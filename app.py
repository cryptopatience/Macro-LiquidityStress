import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fredapi import Fred
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="매크로 유동성(선행지표)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 매크로 유동성 (선행지표) 모니터링")
st.caption("단기유동성으로 금융시장 참여자들(딜러, 헤지펀드)이 체감.즉, 미래에 사용할 수 있는 '연료'가 줄어들고 있음을 미리 보여줍니다.실시간 자금 흐름을 반영 (forward-looking)")

# ============================================================
# 1. 로그인 상태 확인 함수
# ============================================================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 퀀트 대시보드 로그인")
    
    with st.form("credentials"):
        username = st.text_input("아이디 (ID)", key="username")
        password = st.text_input("비밀번호 (Password)", type="password", key="password")
        submit_btn = st.form_submit_button("로그인", type="primary")
    
    if submit_btn:
        try:
            if "passwords" in st.secrets and username in st.secrets["passwords"]:
                if password == st.secrets["passwords"][username]:
                    st.session_state['password_correct'] = True
                    st.rerun()
                else:
                    st.error("😕 비밀번호가 올바르지 않습니다.")
            else:
                st.error("😕 존재하지 않는 아이디입니다.")
        except Exception as e:
            st.error(f"로그인 오류: {str(e)}")
            
    return False

if not check_password():
    st.stop()

# ============================================================
# 2. API 키 설정
# ============================================================
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except KeyError:
    st.error("❌ FRED_API_KEY가 Secrets에 설정되지 않았습니다.")
    st.stop()

# Gemini 설정
GEMINI_AVAILABLE = False
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GEMINI_AVAILABLE = True
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.sidebar.warning("⚠️ Gemini API 키가 없어 AI 분석이 비활성화됩니다.")
except Exception as e:
    st.sidebar.warning(f"⚠️ Gemini 초기화 실패: {str(e)}")

# OpenAI 설정 추가
OPENAI_ENABLED = False
OPENAI_CLIENT = None
try:
    if "OPENAI_API_KEY" in st.secrets:
        from openai import OpenAI
        OPENAI_CLIENT = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        OPENAI_ENABLED = True
except Exception as e:
    st.sidebar.warning(f"⚠️ OpenAI 초기화 실패: {e}")

# 모델 설정
OPENAI_MODEL_CHAT = st.secrets.get("OPENAI_MODEL_CHAT", "gpt-4o")
GEMINI_MODEL_CHAT = "gemini-2.5-flash"

fred = Fred(api_key=FRED_API_KEY)

SERIES_IDS = {
    'RP': 'RPONTSYD',
    'RRP': 'RRPONTSYD',
    'Reserves': 'WRESBAL',
    'SOFR': 'SOFR',
    'IORB': 'IORB'
}

WARNING_LEVELS = {
    'RP': {
        'normal': 20,
        'warning': 30,
        'danger': 50
    },
    'RRP': {
        'danger': 100,
        'warning': 200,
        'normal': 300
    },
    'Reserves': {
        'danger': 3000,
        'warning': 3200
    },
    'Spread': {
        'normal': 10,
        'warning': 20,
        'danger': 100
    }
}

# ============================================================
# 3. 데이터 수집 함수
# ============================================================
@st.cache_data(ttl=3600)
def fetch_data_with_ffill(series_id, start_date, name=""):
    """FRED API에서 데이터를 가져오고 forward fill로 결측치 처리"""
    try:
        data = fred.get_series(series_id, observation_start=start_date)
        if len(data) > 0:
            data = data.ffill()
            return data
        else:
            return pd.Series()
    except Exception as e:
        st.error(f"❌ {name or series_id} 수집 실패: {e}")
        return pd.Series()

@st.cache_data(ttl=3600)
def fetch_liquidity_data(start_date):
    """유동성 데이터를 가져오고 이동평균 계산"""
    
    with st.spinner('📡 FRED API에서 데이터 수집 중...'):
        rp_data = fetch_data_with_ffill('RPONTSYD', start_date, "RP (Repo)")
        rrp_data = fetch_data_with_ffill('RRPONTSYD', start_date, "RRP (Reverse Repo)")
        reserves_data = fetch_data_with_ffill('WRESBAL', start_date, "은행 지준금")
        sofr_data = fetch_data_with_ffill('SOFR', start_date, "SOFR")
        iorb_data = fetch_data_with_ffill('IORB', start_date, "IORB")
        
        all_data = {
            'RP': rp_data,
            'RRP': rrp_data,
            'Reserves': reserves_data,
            'SOFR': sofr_data,
            'IORB': iorb_data
        }
        
        df = pd.DataFrame(all_data)
        df = df.ffill()
        df = df[df.index >= start_date]
        
        df['Spread_bps'] = (df['SOFR'] - df['IORB']) * 100
        
        for col in ['RP', 'RRP', 'Reserves', 'Spread_bps']:
            if col in df.columns:
                df[f'{col}_MA7'] = df[col].rolling(window=7, min_periods=1).mean()
                df[f'{col}_MA30'] = df[col].rolling(window=30, min_periods=1).mean()
                df[f'{col}_MA60'] = df[col].rolling(window=60, min_periods=1).mean()
        
        return df

# ============================================================
# 4. 종합 상태 평가 시스템
# ============================================================
def assess_liquidity_status(df):
    """종합적인 유동성 상태 평가"""
    latest = df.iloc[-1]
    latest_date = df.index[-1].strftime('%Y-%m-%d')
    
    assessments = {}
    
    # RP 평가
    rp_val = latest['RP']
    if rp_val > WARNING_LEVELS['RP']['danger']:
        rp_status = {'level': '🔴 위험', 'score': 0, 'message': 'RP 급증 - 긴급 유동성 수요'}
    elif rp_val > WARNING_LEVELS['RP']['warning']:
        rp_status = {'level': '🟠 경고', 'score': 1, 'message': 'RP 증가 - 단기 자금 수요 상승'}
    elif rp_val > WARNING_LEVELS['RP']['normal']:
        rp_status = {'level': '🟡 주의', 'score': 2, 'message': 'RP 정상 상한 근접'}
    else:
        rp_status = {'level': '🟢 정상', 'score': 3, 'message': 'RP 안정적 수준'}
    assessments['RP'] = rp_status
    
    # RRP 평가
    rrp_val = latest['RRP']
    if rrp_val < WARNING_LEVELS['RRP']['danger']:
        rrp_status = {'level': '🔴 위험', 'score': 0, 'message': 'RRP 극저점 - 시장 현금 부족'}
    elif rrp_val < WARNING_LEVELS['RRP']['warning']:
        rrp_status = {'level': '🟠 경고', 'score': 1, 'message': 'RRP 저점 - 유동성 감소'}
    elif rrp_val < WARNING_LEVELS['RRP']['normal']:
        rrp_status = {'level': '🟡 주의', 'score': 2, 'message': 'RRP 정상 하한 근접'}
    else:
        rrp_status = {'level': '🟢 정상', 'score': 3, 'message': 'RRP 충분한 수준'}
    assessments['RRP'] = rrp_status
    
    # 지준금 평가
    res_val = latest['Reserves']
    if len(df) >= 30:
        res_change_30d = ((latest['Reserves'] - df['Reserves'].iloc[-30]) / df['Reserves'].iloc[-30]) * 100
    else:
        res_change_30d = 0
    
    if res_val < WARNING_LEVELS['Reserves']['danger']:
        res_status = {'level': '🔴 위험', 'score': 0, 'message': f'지준금 위험 수준 (30일 변화: {res_change_30d:.1f}%)'}
    elif res_val < WARNING_LEVELS['Reserves']['warning']:
        res_status = {'level': '🟠 경고', 'score': 1, 'message': f'지준금 감소 추세 (30일 변화: {res_change_30d:.1f}%)'}
    elif res_change_30d < -5:
        res_status = {'level': '🟡 주의', 'score': 2, 'message': f'지준금 급감 (30일 변화: {res_change_30d:.1f}%)'}
    else:
        res_status = {'level': '🟢 정상', 'score': 3, 'message': f'지준금 안정적 (30일 변화: {res_change_30d:.1f}%)'}
    assessments['Reserves'] = res_status
    
    # 스프레드 평가
    spread_val = latest['Spread_bps']
    if spread_val > WARNING_LEVELS['Spread']['danger']:
        spread_status = {'level': '🔴 위험', 'score': 0, 'message': '스프레드 극단 확대 - 유동성 위기'}
    elif spread_val > WARNING_LEVELS['Spread']['warning']:
        spread_status = {'level': '🟠 경고', 'score': 1, 'message': '스프레드 확대 - 자금 조달 압박'}
    elif spread_val > WARNING_LEVELS['Spread']['normal']:
        spread_status = {'level': '🟡 주의', 'score': 2, 'message': '스프레드 정상 상한 근접'}
    else:
        spread_status = {'level': '🟢 정상', 'score': 3, 'message': '스프레드 안정적'}
    assessments['Spread'] = spread_status
    
    # 종합 점수 계산
    total_score = sum(a['score'] for a in assessments.values())
    max_score = 12
    
    # 종합 평가
    if total_score >= 10:
        overall = {
            'status': '🟢 양호',
            'level': 'NORMAL',
            'message': '모든 유동성 지표가 정상 범위입니다.',
            'recommendation': '정상적인 시장 모니터링 유지'
        }
    elif total_score >= 7:
        overall = {
            'status': '🟡 주의',
            'level': 'CAUTION',
            'message': '일부 지표에서 경미한 이상 신호가 감지되었습니다.',
            'recommendation': '시장 동향 면밀히 관찰'
        }
    elif total_score >= 4:
        overall = {
            'status': '🟠 경고',
            'level': 'WARNING',
            'message': '유동성 스트레스 신호가 나타나고 있습니다.',
            'recommendation': '포트폴리오 리스크 관리 강화 필요'
        }
    else:
        overall = {
            'status': '🔴 위험',
            'level': 'DANGER',
            'message': '심각한 유동성 긴장 상태입니다.',
            'recommendation': '긴급 리스크 헤지 조치 권고'
        }
    
    overall['score'] = total_score
    overall['max_score'] = max_score
    
    return {
        'assessments': assessments,
        'overall': overall,
        'latest_values': {
            'RP': latest['RP'],
            'RRP': latest['RRP'],
            'Reserves': latest['Reserves'],
            'Spread': latest['Spread_bps'],
            'SOFR': latest['SOFR'],
            'IORB': latest['IORB']
        },
        'latest_date': latest_date
    }

# ============================================================
# 5. ✨ Enhanced Dual AI Handler (Advanced Chat 추가)
# ============================================================
class EnhancedDualAIHandler:
    @staticmethod
    def generate_liquidity_context(df, assessment):
        """유동성 데이터 컨텍스트 생성"""
        latest = df.iloc[-1]
        last_30d = df.tail(30) if len(df) >= 30 else df
        
        changes = {}
        for col in ['RP', 'RRP', 'Reserves', 'Spread_bps']:
            if len(last_30d) >= 2 and last_30d[col].iloc[0] != 0:
                change = ((latest[col] - last_30d[col].iloc[0]) / last_30d[col].iloc[0]) * 100
                changes[col] = change
            else:
                changes[col] = 0.0
        
        context = f"### 🏦 연준 유동성 분석 데이터 (생성: {datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n"
        
        context += f"**종합 상태:**\n"
        context += f"- 평가: {assessment['overall']['status']} (점수: {assessment['overall']['score']}/{assessment['overall']['max_score']})\n"
        context += f"- 메시지: {assessment['overall']['message']}\n"
        context += f"- 권고: {assessment['overall']['recommendation']}\n\n"
        
        context += f"**주요 지표 (최신: {assessment['latest_date']}):**\n"
        context += f"- RP (Repo): ${latest['RP']:.2f}B (30일 변화: {changes['RP']:+.1f}%) - {assessment['assessments']['RP']['level']}\n"
        context += f"- RRP (Reverse Repo): ${latest['RRP']:.2f}B (30일 변화: {changes['RRP']:+.1f}%) - {assessment['assessments']['RRP']['level']}\n"
        context += f"- 은행 지준금: ${latest['Reserves']:.2f}B (30일 변화: {changes['Reserves']:+.1f}%) - {assessment['assessments']['Reserves']['level']}\n"
        context += f"- SOFR-IORB 스프레드: {latest['Spread_bps']:.2f}bps (30일 변화: {changes['Spread_bps']:+.1f}bps) - {assessment['assessments']['Spread']['level']}\n\n"
        
        context += f"**이동평균:**\n"
        context += f"- RP: MA7={df['RP_MA7'].iloc[-1]:.2f}B, MA30={df['RP_MA30'].iloc[-1]:.2f}B\n"
        context += f"- RRP: MA7={df['RRP_MA7'].iloc[-1]:.2f}B, MA30={df['RRP_MA30'].iloc[-1]:.2f}B\n"
        context += f"- 지준금: MA7={df['Reserves_MA7'].iloc[-1]:.2f}B, MA30={df['Reserves_MA30'].iloc[-1]:.2f}B\n"
        context += f"- 스프레드: MA7={df['Spread_bps_MA7'].iloc[-1]:.2f}bps, MA30={df['Spread_bps_MA30'].iloc[-1]:.2f}bps\n"
        
        return context

    @staticmethod
    def query_advanced_chat(prompt, context, model_choice, chat_history):
        """
        ✨ Advanced Chat: 유동성 데이터 + 대화 히스토리를 결합하여 
        AI가 현재 상황을 인지한 상태로 답변
        """
        system_instruction = f"""
        당신은 연준 유동성 정책, 매크로 경제, 시장 리스크 관리 전문가입니다.
        
        [현재 실시간 유동성 분석 데이터]
        {context}
        
        [지시사항]
        1. 위 [유동성 분석 데이터]의 수치(RP, RRP, 지준금, 스프레드)를 근거로 답변하세요.
        2. 유동성 긴축/완화가 주식, 채권, 금, 암호화폐 등 자산군에 미치는 영향을 설명하세요.
        3. 2008 금융위기, 2020 코로나 위기 등 과거 유사 패턴과 비교하여 인사이트를 제공하세요.
        4. 감정적 희망보다는 통계와 역사적 패턴에 기반한 객관적 뷰를 제시하세요.
        5. 한국어로 간결하고 명확하게 답변하세요.
        """

        # Gemini 로직
        if model_choice == "Gemini":
            if not GEMINI_AVAILABLE: 
                return "⚠️ Gemini API 설정이 필요합니다."
            try:
                model = genai.GenerativeModel(GEMINI_MODEL_CHAT)
                
                full_prompt = system_instruction + "\n\n[이전 대화 내역]\n"
                for msg in chat_history[-10:]:
                    role_label = "User" if msg['role'] == 'user' else "AI"
                    full_prompt += f"{role_label}: {msg['content']}\n"
                
                full_prompt += f"\n[User 질문]: {prompt}\n[AI 답변]:"
                
                response = model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                return f"⚠️ Gemini 오류: {str(e)}"

        # OpenAI 로직
        else: 
            if not OPENAI_ENABLED: 
                return "⚠️ OpenAI API 설정이 필요합니다."
            try:
                messages = [{"role": "system", "content": system_instruction}]
                messages.extend(chat_history[-6:])
                messages.append({"role": "user", "content": prompt})
                
                response = OPENAI_CLIENT.chat.completions.create(
                    model=OPENAI_MODEL_CHAT,
                    messages=messages,
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"⚠️ OpenAI 오류: {str(e)}"

# ============================================================
# 5-1. Gemini AI 분석 함수 (종합 분석)
# ============================================================
def generate_gemini_analysis(df, assessment):
    """Gemini 2.5 Flash를 사용한 종합 AI 분석"""
    
    if not GEMINI_AVAILABLE:
        return """
### ⚠️ Gemini AI 분석을 사용할 수 없습니다

Gemini API 키가 설정되지 않았습니다. 

**Gemini API 키 설정 방법:**
1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 발급
2. Streamlit Cloud의 App settings → Secrets에 추가:
```toml
   GEMINI_API_KEY = "your_key_here"
```
3. 앱 재시작
"""
    
    latest = df.iloc[-1]
    last_30d = df.tail(30) if len(df) >= 30 else df
    
    changes = {}
    for col in ['RP', 'RRP', 'Reserves', 'Spread_bps']:
        if len(last_30d) >= 2 and last_30d[col].iloc[0] != 0:
            change = ((latest[col] - last_30d[col].iloc[0]) / last_30d[col].iloc[0]) * 100
            changes[col] = change
        else:
            changes[col] = 0.0
    
    prompt = f"""
당신은 연준 유동성 정책 및 거시경제 전문가입니다. 다음 데이터를 분석하고 한국어로 상세한 인사이트를 제공해주세요.

## 현재 유동성 지표 (최신 날짜: {assessment['latest_date']})

### 주요 지표:
- RP (Repo): ${latest['RP']:.2f}B (30일 변화: {changes.get('RP', 0):+.1f}%)
- RRP (Reverse Repo): ${latest['RRP']:.2f}B (30일 변화: {changes.get('RRP', 0):+.1f}%)
- 은행 지준금: ${latest['Reserves']:.2f}B (30일 변화: {changes.get('Reserves', 0):+.1f}%)
- SOFR-IORB 스프레드: {latest['Spread_bps']:.2f}bps (30일 변화: {changes.get('Spread_bps', 0):+.1f}bps)

### 종합 평가:
- 상태: {assessment['overall']['status']}
- 점수: {assessment['overall']['score']}/{assessment['overall']['max_score']}
- 평가: {assessment['overall']['message']}

### 개별 지표 상태:
- RP: {assessment['assessments']['RP']['level']} - {assessment['assessments']['RP']['message']}
- RRP: {assessment['assessments']['RRP']['level']} - {assessment['assessments']['RRP']['message']}
- 지준금: {assessment['assessments']['Reserves']['level']} - {assessment['assessments']['Reserves']['message']}
- 스프레드: {assessment['assessments']['Spread']['level']} - {assessment['assessments']['Spread']['message']}

## 분석 요청사항:

1. **현재 유동성 상황 종합 평가** (3-4문장)
2. **주요 리스크 및 기회** (5-6개 bullet points)
3. **향후 전망 및 시나리오** (3가지 시나리오, 각 확률 포함)
4. **투자 전략 제언** (구체적인 자산배분·리스크 관리·모니터링 포인트)

간결하고 실용적인 분석을 부탁드립니다. 전문 용어 사용 시 간단한 설명을 추가해주세요.
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        generation_config = {
            'max_output_tokens': 2048,
            'temperature': 0.7,
        }
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return f"""
### ⚠️ Gemini API 할당량 초과

현재 Gemini API의 무료 tier 할당량을 초과했습니다.

**해결 방법:**
1. **잠시 대기**: 10-60분 후 다시 시도
2. **할당량 확인**: [사용량 확인](https://ai.dev/usage?tab=rate-limit)

**현재 상태 요약 (수동 분석):**
- 종합 평가: {assessment['overall']['status']}
- RP: {assessment['assessments']['RP']['level']}
- RRP: {assessment['assessments']['RRP']['level']}
- 지준금: {assessment['assessments']['Reserves']['level']}
- 스프레드: {assessment['assessments']['Spread']['level']}

**권고사항**: {assessment['overall']['recommendation']}
"""
        return f"⚠️ AI 분석 생성 중 오류 발생: {error_msg}"

# ============================================================
# 5-2. Gemini AI 분석 함수 (개별 지표 분석)
# ============================================================
def generate_gemini_single_indicator(df, assessment, indicator, depth="기본"):
    """Gemini 2.5 Flash를 사용한 개별 지표 AI 분석"""
    
    if not GEMINI_AVAILABLE:
        return """
### ⚠️ Gemini AI 분석을 사용할 수 없습니다

Gemini API 키가 설정되지 않았습니다.
"""
    
    if indicator == "Spread":
        col = "Spread_bps"
        display_name = "SOFR - IORB 스프레드"
        unit = "bps"
        key_for_assessment = "Spread"
    else:
        col = indicator
        display_name = {
            "RP": "RP (Repo)",
            "RRP": "RRP (Reverse Repo)",
            "Reserves": "은행 지준금"
        }.get(indicator, indicator)
        unit = "B"
        key_for_assessment = indicator
    
    latest = df.iloc[-1]
    last_30d = df.tail(30) if len(df) >= 30 else df
    
    if len(df) >= 7 and df[col].iloc[-7] != 0:
        change_7d = ((df[col].iloc[-1] - df[col].iloc[-7]) / df[col].iloc[-7]) * 100
    else:
        change_7d = 0.0
    
    if len(last_30d) >= 2 and last_30d[col].iloc[0] != 0:
        change_30d = ((last_30d[col].iloc[-1] - last_30d[col].iloc[0]) / last_30d[col].iloc[0]) * 100
    else:
        change_30d = 0.0
    
    ma7 = df[f"{col}_MA7"].iloc[-1]
    ma30 = df[f"{col}_MA30"].iloc[-1]
    ma60 = df[f"{col}_MA60"].iloc[-1]
    
    status_info = assessment["assessments"][key_for_assessment]
    
    prompt = f"""
당신은 연준 유동성 지표 전문가입니다.
다음 하나의 지표에 대해서만 깊이 있게 분석해 주세요. 한국어로 답변해 주세요.

## 분석 지표 정보
- 지표 이름: {display_name}
- 최신 값: {latest[col]:.2f}{unit}
- 7일 변화율: {change_7d:+.1f}%
- 30일 변화율: {change_30d:+.1f}%
- MA7: {ma7:.2f}{unit}, MA30: {ma30:.2f}{unit}, MA60: {ma60:.2f}{unit}
- 현재 상태: {status_info['level']} - {status_info['message']}
- 전체 유동성 종합 상태: {assessment['overall']['status']} (점수 {assessment['overall']['score']}/{assessment['overall']['max_score']})

## 분석 요청 항목
1. 현재 수준과 최근 1~3개월 추세 요약
2. 이동평균(MA7/30/60) 관점에서 본 단기 vs 중기 추세
3. 경고/위험 레벨과의 거리 및 스트레스 정도 평가
4. 과거 유사 수준에서 나타났던 전형적인 시장 패턴
5. 투자자 관점에서의 리스크 요인과 잠재적 기회
6. 앞으로 주시해야 할 트리거 레벨과 대응 전략

### 분석 깊이 모드: {depth}
- '요약' 모드: 각 항목을 1~2문장으로 간단히 요약
- '기본' 모드: 각 항목을 2~3문장 정도로 설명
- '딥다이브' 모드: 각 항목을 자세히 설명하고 필요시 bullet을 사용

위 모드에 맞추어 답변의 상세도를 조절해 주세요.
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        generation_config = {
            'max_output_tokens': 2048 if depth == "딥다이브" else 1024,
            'temperature': 0.7,
        }
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return "⚠️ API 할당량 초과. 잠시 후 다시 시도하세요."
        return f"⚠️ AI 개별 지표 분석 생성 중 오류 발생: {error_msg}"

# ============================================================
# 6. 메인 차트 생성 함수
# ============================================================
def create_main_chart(df, assessment):
    """메인 대시보드 차트 생성"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            f'📈 RP (Repo) - 현재: ${assessment["latest_values"]["RP"]:.2f}B | 상태: {assessment["assessments"]["RP"]["level"]}',
            f'📉 RRP (Reverse Repo) - 현재: ${assessment["latest_values"]["RRP"]:.2f}B | 상태: {assessment["assessments"]["RRP"]["level"]}',
            f'💰 은행 지준금 - 현재: ${assessment["latest_values"]["Reserves"]:.2f}B | 상태: {assessment["assessments"]["Reserves"]["level"]}',
            f'🔴 SOFR-IORB 스프레드 - 현재: {assessment["latest_values"]["Spread"]:.2f}bps | 상태: {assessment["assessments"]["Spread"]["level"]}'
        ),
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )
    
    # RP 차트
    fig.add_trace(go.Scatter(x=df.index, y=df['RP'], name='RP', line=dict(color='#20c997', width=2.5), legendgroup='rp'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RP_MA7'], name='MA7', line=dict(color='#1abc9c', width=1.5, dash='dot'), legendgroup='rp'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RP_MA30'], name='MA30', line=dict(color='#16a085', width=1.5, dash='dash'), legendgroup='rp'), row=1, col=1)
    fig.add_hline(y=WARNING_LEVELS['RP']['normal'], line_dash="dot", line_color="yellow", annotation_text="주의 (20B)", annotation_position="right", row=1, col=1)
    fig.add_hline(y=WARNING_LEVELS['RP']['warning'], line_dash="dash", line_color="orange", annotation_text="경고 (30B)", annotation_position="right", row=1, col=1)
    fig.add_hline(y=WARNING_LEVELS['RP']['danger'], line_dash="solid", line_color="red", annotation_text="위험 (50B)", annotation_position="right", row=1, col=1)
    
    # RRP 차트
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP'], name='RRP', line=dict(color='#3498db', width=2.5), legendgroup='rrp'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP_MA7'], name='MA7', line=dict(color='#5dade2', width=1.5, dash='dot'), legendgroup='rrp'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP_MA30'], name='MA30', line=dict(color='#2980b9', width=1.5, dash='dash'), legendgroup='rrp'), row=2, col=1)
    fig.add_hline(y=WARNING_LEVELS['RRP']['danger'], line_dash="solid", line_color="red", annotation_text="위험 (100B)", annotation_position="right", row=2, col=1)
    fig.add_hline(y=WARNING_LEVELS['RRP']['warning'], line_dash="dash", line_color="orange", annotation_text="경고 (200B)", annotation_position="right", row=2, col=1)
    fig.add_hline(y=WARNING_LEVELS['RRP']['normal'], line_dash="dot", line_color="yellow", annotation_text="주의 (300B)", annotation_position="right", row=2, col=1)
    
    # 지준금 차트
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves'], name='Reserves', line=dict(color='#f39c12', width=2.5), legendgroup='reserves'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves_MA7'], name='MA7', line=dict(color='#f8b739', width=1.5, dash='dot'), legendgroup='reserves'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves_MA30'], name='MA30', line=dict(color='#e67e22', width=1.5, dash='dash'), legendgroup='reserves'), row=3, col=1)
    fig.add_hline(y=WARNING_LEVELS['Reserves']['danger'], line_dash="solid", line_color="red", annotation_text="위험 (3,000B)", annotation_position="right", row=3, col=1)
    fig.add_hline(y=WARNING_LEVELS['Reserves']['warning'], line_dash="dash", line_color="orange", annotation_text="경고 (3,200B)", annotation_position="right", row=3, col=1)
    
    # 스프레드 차트
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps'], name='Spread', line=dict(color='#e74c3c', width=2.5), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)', legendgroup='spread'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps_MA7'], name='MA7', line=dict(color='#ec7063', width=1.5, dash='dot'), legendgroup='spread'), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps_MA30'], name='MA30', line=dict(color='#c0392b', width=1.5, dash='dash'), legendgroup='spread'), row=4, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=4, col=1)
    fig.add_hline(y=WARNING_LEVELS['Spread']['normal'], line_dash="dot", line_color="yellow", annotation_text="주의 (10bps)", annotation_position="right", row=4, col=1)
    fig.add_hline(y=WARNING_LEVELS['Spread']['warning'], line_dash="dash", line_color="orange", annotation_text="경고 (20bps)", annotation_position="right", row=4, col=1)
    fig.add_hline(y=WARNING_LEVELS['Spread']['danger'], line_dash="solid", line_color="red", annotation_text="위험 (100bps)", annotation_position="right", row=4, col=1)
    
    fig.update_layout(
        height=1400,
        title_text=f"🏦 연준 유동성 스트레스 모니터링 대시보드<br><sub>종합 평가: {assessment['overall']['status']} ({assessment['overall']['score']}/{assessment['overall']['max_score']}점)</sub>",
        title_font_size=20,
        showlegend=True,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="십억 달러 ($B)", row=1, col=1)
    fig.update_yaxes(title_text="십억 달러 ($B)", row=2, col=1)
    fig.update_yaxes(title_text="십억 달러 ($B)", row=3, col=1)
    fig.update_yaxes(title_text="베이시스 포인트 (bps)", row=4, col=1)
    fig.update_xaxes(title_text="날짜", row=4, col=1)
    
    return fig

# ============================================================
# 7. 이동평균 교차 차트
# ============================================================
def create_ma_crossover_chart(df):
    """이동평균 교차 분석 차트"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('RP 이동평균 교차', 'RRP 이동평균 교차', '지준금 이동평균 교차', '스프레드 이동평균 교차')
    )
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RP'], name='실제값', line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RP_MA7'], name='7일', line=dict(dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RP_MA30'], name='30일', line=dict(dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['RP_MA60'], name='60일', line=dict(dash='longdash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP'], name='실제값', line=dict(width=1), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP_MA7'], name='7일', line=dict(dash='dot'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP_MA30'], name='30일', line=dict(dash='dash'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['RRP_MA60'], name='60일', line=dict(dash='longdash'), showlegend=False), row=1, col=2)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves'], name='실제값', line=dict(width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves_MA7'], name='7일', line=dict(dash='dot'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves_MA30'], name='30일', line=dict(dash='dash'), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Reserves_MA60'], name='60일', line=dict(dash='longdash'), showlegend=False), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps'], name='실제값', line=dict(width=1), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps_MA7'], name='7일', line=dict(dash='dot'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps_MA30'], name='30일', line=dict(dash='dash'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['Spread_bps_MA60'], name='60일', line=dict(dash='longdash'), showlegend=False), row=2, col=2)
    
    fig.update_layout(height=800, title_text="📊 이동평균 교차 분석", showlegend=True)
    
    return fig

# ============================================================
# 8. 메인 앱
# ============================================================
def main():
    # 사이드바
    st.sidebar.header("⚙️ 분석 설정")
    
    # API 상태 표시
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if GEMINI_AVAILABLE:
            st.success("✅ Gemini")
        else:
            st.error("❌ Gemini")
    with col2:
        if OPENAI_ENABLED:
            st.success("✅ OpenAI")
        else:
            st.error("❌ OpenAI")
    
    st.sidebar.markdown("---")
    
    period_options = {
        "최근 1년": 365,
        "최근 2년": 730,
        "최근 5년": 1825,
        "2008년 금융위기 이후 (2007-)": None,
        "2000년 이후 (닷컴 버블 포함)": None,
        "사용자 정의": "custom"
    }
    
    selected_period = st.sidebar.selectbox("📅 분석 기간 선택", list(period_options.keys()), index=0)
    
    if selected_period == "2008년 금융위기 이후 (2007-)":
        start_date = '2007-01-01'
        period_name = "2008년 금융위기 이후"
    elif selected_period == "2000년 이후 (닷컴 버블 포함)":
        start_date = '2000-01-01'
        period_name = "2000년 이후"
    elif selected_period == "사용자 정의":
        custom_date = st.sidebar.date_input("시작 날짜 선택", value=datetime.now() - timedelta(days=365))
        start_date = custom_date.strftime('%Y-%m-%d')
        period_name = f"{start_date}부터"
    else:
        lookback_days = period_options[selected_period]
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        period_name = selected_period
    
    st.sidebar.success(f"✅ 선택된 기간: {period_name}")
    
    if st.sidebar.button("🔄 데이터 새로고침", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # 데이터 로드
    try:
        df = fetch_liquidity_data(start_date)
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        return
    
    if df.empty:
        st.error("❌ 데이터를 불러올 수 없습니다.")
        return
    
    assessment = assess_liquidity_status(df)
    
    # ✨ 컨텍스트 생성 및 저장
    if 'liquidity_context' not in st.session_state:
        st.session_state['liquidity_context'] = EnhancedDualAIHandler.generate_liquidity_context(df, assessment)
    
    # 상단 메트릭
    st.markdown("### 📊 현재 유동성 지표")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RP (Repo)", f"${assessment['latest_values']['RP']:.2f}B", help="연준의 단기 대출")
        st.markdown(f"**상태:** {assessment['assessments']['RP']['level']}")
    
    with col2:
        st.metric("RRP (Reverse Repo)", f"${assessment['latest_values']['RRP']:.2f}B", help="시장의 여유자금")
        st.markdown(f"**상태:** {assessment['assessments']['RRP']['level']}")
    
    with col3:
        st.metric("은행 지준금", f"${assessment['latest_values']['Reserves']:.2f}B", help="은행의 즉시 사용 가능 현금")
        st.markdown(f"**상태:** {assessment['assessments']['Reserves']['level']}")
    
    with col4:
        st.metric("SOFR-IORB 스프레드", f"{assessment['latest_values']['Spread']:.2f}bps", help="시장금리와 기준금리 차이")
        st.markdown(f"**상태:** {assessment['assessments']['Spread']['level']}")
    
    # 종합 평가
    st.markdown("---")
    st.markdown("### 🎯 종합 평가")
    
    status_color = {'🟢 양호': 'green', '🟡 주의': 'orange', '🟠 경고': 'orange', '🔴 위험': 'red'}
    
    st.markdown(
        f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {status_color.get(assessment['overall']['status'], 'gray')}20; border-left: 5px solid {status_color.get(assessment['overall']['status'], 'gray')}'>
            <h2>{assessment['overall']['status']}</h2>
            <p style='font-size: 18px;'><strong>점수:</strong> {assessment['overall']['score']}/{assessment['overall']['max_score']}</p>
            <p style='font-size: 16px;'>{assessment['overall']['message']}</p>
            <p style='font-size: 16px;'><strong>📌 권고사항:</strong> {assessment['overall']['recommendation']}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 메인 차트
    st.markdown("---")
    st.markdown("### 📈 유동성 지표 추이")
    main_chart = create_main_chart(df, assessment)
    st.plotly_chart(main_chart, use_container_width=True)
    
    # 탭 - Advanced Chat 추가
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 개별 지표 분석", 
        "📈 이동평균 교차", 
        "🤖 AI 종합 분석", 
        "💬 Advanced Chat",
        "📖 해석 가이드"
    ])
    
    with tab1:
        st.markdown("### 개별 지표 상세 분석")
        
        for indicator, data in assessment['assessments'].items():
            with st.expander(f"{indicator} - {data['level']}", expanded=False):
                col_name = 'Spread_bps' if indicator == 'Spread' else indicator
                
                current_val = assessment['latest_values']['Spread' if indicator == 'Spread' else indicator]
                ma7 = df[f'{col_name}_MA7'].iloc[-1]
                ma30 = df[f'{col_name}_MA30'].iloc[-1]
                ma60 = df[f'{col_name}_MA60'].iloc[-1]
                
                c1, c2, c3, c4 = st.columns(4)
                
                unit = "bps" if indicator == 'Spread' else "B"
                prefix = "" if indicator == 'Spread' else "$"
                
                with c1:
                    st.metric("현재값", f"{prefix}{current_val:.2f}{unit}")
                with c2:
                    st.metric("7일 이평", f"{prefix}{ma7:.2f}{unit}")
                with c3:
                    st.metric("30일 이평", f"{prefix}{ma30:.2f}{unit}")
                with c4:
                    st.metric("60일 이평", f"{prefix}{ma60:.2f}{unit}")
                
                st.markdown(f"**평가:** {data['message']}")
    
    with tab2:
        st.markdown("### 이동평균 교차 분석")
        ma_chart = create_ma_crossover_chart(df)
        st.plotly_chart(ma_chart, use_container_width=True)
    
    with tab3:
        st.markdown("### 🤖 Gemini AI 분석")
        
        analysis_mode = st.radio("분석 모드 선택", ["종합 분석", "개별 지표 분석"], horizontal=True)
        
        if analysis_mode == "종합 분석":
            if st.button("🚀 종합 AI 분석 실행", type="primary"):
                with st.spinner("🧠 Gemini가 종합 분석 중..."):
                    ai_analysis = generate_gemini_analysis(df, assessment)
                    st.session_state['comprehensive_analysis'] = ai_analysis
            
            if 'comprehensive_analysis' in st.session_state:
                st.markdown(st.session_state['comprehensive_analysis'])
                st.download_button(
                    "📥 종합 분석 다운로드",
                    st.session_state['comprehensive_analysis'],
                    f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    "text/markdown"
                )
        
        else:  # 개별 지표 분석
            st.markdown("#### 분석할 지표와 분석 깊이를 선택하세요.")
            indicator = st.selectbox("분석할 지표 선택", ["RP", "RRP", "Reserves", "Spread"])
            depth = st.select_slider("분석 깊이 선택", options=["요약", "기본", "딥다이브"], value="기본")
            
            if st.button("🔍 선택 지표 AI 분석 실행", type="primary"):
                with st.spinner(f"🧠 Gemini가 {indicator} 분석 중..."):
                    ai_single = generate_gemini_single_indicator(df, assessment, indicator, depth)
                    st.session_state['single_analysis'] = ai_single
                    st.session_state['single_indicator'] = indicator
            
            if 'single_analysis' in st.session_state:
                st.markdown(st.session_state['single_analysis'])
                st.download_button(
                    "📥 개별 분석 다운로드",
                    st.session_state['single_analysis'],
                    f"single_analysis_{st.session_state.get('single_indicator', 'indicator')}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    "text/markdown"
                )
    
    # ✨ 탭 4: Advanced AI 채팅
    with tab4:
        st.header("💬 Advanced Quant Chatbot")
        
        # 레이아웃: 채팅창(왼쪽) vs 제어패널(오른쪽)
        col_chat, col_ctrl = st.columns([3, 1])
        
        # 1. 오른쪽 제어 패널
        with col_ctrl:
            st.markdown("### 🎛️ 제어 패널")
            
            available_models = []
            if OPENAI_ENABLED: available_models.append("OpenAI")
            if GEMINI_AVAILABLE: available_models.append("Gemini")
            
            if not available_models:
                st.error("API 키가 없습니다.")
                model_choice = None
            else:
                model_choice = st.radio("🧠 모델 선택", available_models, index=0)
            
            st.info(f"**모드 특징**\n- Gemini: 거시경제/종합해석\n- OpenAI: 수치분석/논리")
            
            st.markdown("---")
            if st.button("🧹 대화 지우기", use_container_width=True):
                st.session_state.advanced_chat_messages = []
                st.rerun()
            
            with st.expander("데이터 컨텍스트 확인"):
                st.caption(st.session_state.get('liquidity_context', '데이터 분석을 먼저 실행하세요.'))

        # 2. 왼쪽 채팅창
        with col_chat:
            # 초기화
            if "advanced_chat_messages" not in st.session_state:
                st.session_state.advanced_chat_messages = []

            # 대화 기록 표시
            for msg in st.session_state.advanced_chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # ✨ 빠른 질문 버튼 (Quick Replies) - 유동성 맞춤형
            quick_questions = [
                "💰 현재 유동성 상태는 위험한가요?",
                "📊 RRP 급감이 주식시장에 미치는 영향은?",
                "🔴 지준금 감소 시 어떤 자산이 유리한가요?",
                "⚠️ 스프레드 확대는 무엇을 의미하나요?",
                "💡 지금 추천하는 리스크 관리 전략은?"
            ]
            
            # 버튼을 가로로 배열
            btn_cols = st.columns(len(quick_questions))
            triggered_prompt = None
            
            for i, question in enumerate(quick_questions):
                if btn_cols[i].button(question, key=f"quick_liq_btn_{i}", use_container_width=True):
                    triggered_prompt = question

            # 3. 입력 처리 (채팅창 입력 OR 버튼 클릭)
            user_input = st.chat_input("질문을 입력하세요 (예: RP가 50B를 넘으면 어떻게 대응해야 할까?)")
            
            # 버튼이 눌렸거나, 채팅창에 입력이 들어오면 실행
            final_prompt = triggered_prompt if triggered_prompt else user_input

            if final_prompt:
                if not model_choice:
                    st.error("AI 모델을 선택해주세요.")
                else:
                    # 사용자 메시지 표시 및 저장
                    st.chat_message("user").markdown(final_prompt)
                    st.session_state.advanced_chat_messages.append({"role": "user", "content": final_prompt})

                    # AI 응답 생성
                    with st.chat_message("assistant"):
                        with st.spinner(f"🧠 {model_choice}가 유동성 데이터를 분석 중입니다..."):
                            context = st.session_state.get('liquidity_context', "")
                            
                            response = EnhancedDualAIHandler.query_advanced_chat(
                                prompt=final_prompt,
                                context=context,
                                model_choice=model_choice,
                                chat_history=st.session_state.advanced_chat_messages
                            )
                            
                            st.markdown(response)
                            st.session_state.advanced_chat_messages.append({"role": "assistant", "content": response})
    
    with tab5:
        st.markdown("""
        ### 📖 유동성 지표 해석 가이드
        
        #### 1. RP (Repo) - 환매조건부채권
        - **의미**: 은행들이 연준에서 단기 현금을 빌리는 거래
        - **RP 증가** → 은행의 유동성 부족 → 시장 긴장 신호
        - **정상 범위**: 20B 이하
        - **경고 수준**: 30B 초과
        - **위험 수준**: 50B 초과
        
        #### 2. RRP (Reverse Repo) - 역환매조건부채권
        - **의미**: 시장의 여유자금이 연준에 맡겨지는 거래
        - **RRP 감소** → 시장 현금 부족 → 유동성 긴장
        - **정상 범위**: 300B 이상
        - **경고 수준**: 200B 미만
        - **위험 수준**: 100B 미만
        
        #### 3. 은행 지준금 (Reserves)
        - **의미**: 은행이 연준에 예치한 즉시 사용 가능 현금
        - **지준금 감소** → 은행 대출 여력 축소 → 금융 불안정
        - **경고 수준**: 3,200B 미만
        - **위험 수준**: 3,000B 미만
        
        #### 4. SOFR-IORB 스프레드
        - **의미**: 시장금리(SOFR)와 기준금리(IORB)의 차이
        - **스프레드 확대** → 자금 조달 비용 상승 → 유동성 프리미엄
        - **정상 범위**: 10bps 이하
        - **경고 수준**: 20bps 초과
        - **위험 수준**: 100bps 초과
        
        ---
        
        #### 💡 2008년 금융위기와의 비교
        - **2008년 패턴**: RP 급증 + RRP 고갈 + 지준금 급감 + 스프레드 폭발
        - 장기 데이터(2007년 이후)를 선택하면 위기 시기와 비교 분석 가능
        
        ---
        
        #### 📌 투자 시사점
        - **양호(🟢)**: 정상적인 포트폴리오 운용
        - **주의(🟡)**: 리스크 자산 비중 검토
        - **경고(🟠)**: 방어적 포지션 강화
        - **위험(🔴)**: 현금 비중 확대 및 헤지 전략
        """)
    
    # 위험 신호
    st.markdown("---")
    st.markdown("### 🚨 위험 신호 분석")
    
    risk_signals = []
    
    for indicator in ['RP', 'RRP', 'Reserves', 'Spread']:
        if assessment['assessments'][indicator]['score'] <= 1:
            risk_signals.append(f"⚠️ **{indicator} 위험**: {assessment['assessments'][indicator]['message']}")
    
    if len(df) >= 7:
        rp_change_7d = ((df['RP'].iloc[-1] - df['RP'].iloc[-7]) / df['RP'].iloc[-7]) * 100 if df['RP'].iloc[-7] != 0 else 0
        rrp_change_7d = ((df['RRP'].iloc[-1] - df['RRP'].iloc[-7]) / df['RRP'].iloc[-7]) * 100 if df['RRP'].iloc[-7] != 0 else 0
        
        if abs(rp_change_7d) > 50:
            risk_signals.append(f"🔥 **RP 급변동**: 7일 변화율 {rp_change_7d:+.1f}%")
        
        if abs(rrp_change_7d) > 30:
            risk_signals.append(f"🔥 **RRP 급변동**: 7일 변화율 {rrp_change_7d:+.1f}%")
    
    if risk_signals:
        for signal in risk_signals:
            st.warning(signal)
    else:
        st.success("✅ 현재 심각한 위험 신호 없음")
    
    # 데이터 다운로드
    st.markdown("---")
    st.markdown("### 💾 데이터 다운로드")
    
    col1_d, col2_d = st.columns(2)
    
    with col1_d:
        csv_data = df.to_csv()
        st.download_button(
            "📊 전체 데이터 다운로드 (CSV)",
            csv_data,
            f"liquidity_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with col2_d:
        report = f"""
# 연준 유동성 스트레스 모니터링 리포트

## 분석 정보
- 분석 시점: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 데이터 기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}
- 분석 범위: {period_name}

## 종합 평가
- 상태: {assessment['overall']['status']}
- 점수: {assessment['overall']['score']}/{assessment['overall']['max_score']}
- 평가: {assessment['overall']['message']}
- 권고사항: {assessment['overall']['recommendation']}

## 개별 지표
### RP (Repo)
- 현재값: ${assessment['latest_values']['RP']:.2f}B
- 상태: {assessment['assessments']['RP']['level']}
- 평가: {assessment['assessments']['RP']['message']}

### RRP (Reverse Repo)
- 현재값: ${assessment['latest_values']['RRP']:.2f}B
- 상태: {assessment['assessments']['RRP']['level']}
- 평가: {assessment['assessments']['RRP']['message']}

### 은행 지준금
- 현재값: ${assessment['latest_values']['Reserves']:.2f}B
- 상태: {assessment['assessments']['Reserves']['level']}
- 평가: {assessment['assessments']['Reserves']['message']}

### SOFR-IORB 스프레드
- 현재값: {assessment['latest_values']['Spread']:.2f}bps
- 상태: {assessment['assessments']['Spread']['level']}
- 평가: {assessment['assessments']['Spread']['message']}

## 위험 신호
{chr(10).join(risk_signals) if risk_signals else '✅ 현재 심각한 위험 신호 없음'}
"""
        
        st.download_button(
            "📄 종합 리포트 다운로드 (TXT)",
            report,
            f"liquidity_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            "text/plain"
        )
    
    # 푸터
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
            <p>🏦 연준 유동성 스트레스 모니터링 대시보드 v3.0</p>
            <p>데이터 출처: FRED (Federal Reserve Economic Data) | AI: Gemini 2.5 Flash + OpenAI GPT-4</p>
            <p>⚠️ 본 분석은 투자 권유가 아니며, 참고 목적으로만 활용하시기 바랍니다.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

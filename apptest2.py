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
GEMINI_MODEL_CHAT = "gemini-2.0-flash-exp"

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
# 5-1. Gemini AI 분석 함수 (종합 분석 - 기본/요약 모드)
# ============================================================
def generate_gemini_analysis(df, assessment, depth="기본"):
    """Gemini 2.0 Flash를 사용한 종합 AI 분석 (기본/요약 모드)"""
    
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
    
    # 깊이별 프롬프트
    if depth == "요약":
        prompt = f"""
당신은 연준 유동성 정책 전문가입니다. **매우 간결하게** 핵심만 요약해주세요.

## 현재 유동성 지표 (최신: {assessment['latest_date']})

### 주요 지표:
- RP: ${latest['RP']:.2f}B (30일 변화: {changes.get('RP', 0):+.1f}%)
- RRP: ${latest['RRP']:.2f}B (30일 변화: {changes.get('RRP', 0):+.1f}%)
- 지준금: ${latest['Reserves']:.2f}B (30일 변화: {changes.get('Reserves', 0):+.1f}%)
- 스프레드: {latest['Spread_bps']:.2f}bps

### 종합: {assessment['overall']['status']} ({assessment['overall']['score']}/12점)

## 분석 요청 (각 항목 1-2문장):
1. **현재 유동성 상황** (2문장)
2. **핵심 리스크 3가지** (각 1줄)
3. **투자 전략** (2문장)

간결하고 명확하게 작성하세요.
"""
        max_tokens = 512
    
    else:  # 기본
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
        max_tokens = 2048
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        generation_config = {
            'max_output_tokens': max_tokens,
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
# 5-1-1. ✨ NEW: Gemini AI Deep Dive 종합 분석
# ============================================================
def generate_gemini_analysis_deep_dive(df, assessment):
    """Gemini 2.0 Flash를 사용한 종합 AI Deep Dive 분석"""
    
    if not GEMINI_AVAILABLE:
        return "⚠️ Gemini API가 설정되지 않았습니다."
    
    latest = df.iloc[-1]
    last_30d = df.tail(30) if len(df) >= 30 else df
    last_90d = df.tail(90) if len(df) >= 90 else df
    
    # 변화율 계산
    changes = {}
    for col in ['RP', 'RRP', 'Reserves', 'Spread_bps']:
        if len(last_30d) >= 2 and last_30d[col].iloc[0] != 0:
            change_30d = ((latest[col] - last_30d[col].iloc[0]) / last_30d[col].iloc[0]) * 100
            changes[f'{col}_30d'] = change_30d
        else:
            changes[f'{col}_30d'] = 0.0
        
        if len(last_90d) >= 2 and last_90d[col].iloc[0] != 0:
            change_90d = ((latest[col] - last_90d[col].iloc[0]) / last_90d[col].iloc[0]) * 100
            changes[f'{col}_90d'] = change_90d
        else:
            changes[f'{col}_90d'] = 0.0
    
    prompt = f"""
당신은 20년 경력의 연준 유동성 정책, 거시경제, 금융시장 전문가입니다. **매우 상세하고 심층적인 종합 분석**을 제공해주세요.

## 현재 유동성 지표 (최신: {assessment['latest_date']})

### 주요 지표:
- RP (Repo): ${latest['RP']:.2f}B (30일 변화: {changes.get('RP_30d', 0):+.1f}%, 90일 변화: {changes.get('RP_90d', 0):+.1f}%)
- RRP (Reverse Repo): ${latest['RRP']:.2f}B (30일 변화: {changes.get('RRP_30d', 0):+.1f}%, 90일 변화: {changes.get('RRP_90d', 0):+.1f}%)
- 은행 지준금: ${latest['Reserves']:.2f}B (30일 변화: {changes.get('Reserves_30d', 0):+.1f}%, 90일 변화: {changes.get('Reserves_90d', 0):+.1f}%)
- SOFR-IORB 스프레드: {latest['Spread_bps']:.2f}bps (30일 변화: {changes.get('Spread_bps_30d', 0):+.1f}bps, 90일 변화: {changes.get('Spread_bps_90d', 0):+.1f}bps)

### 종합 평가:
- 상태: {assessment['overall']['status']} (점수: {assessment['overall']['score']}/{assessment['overall']['max_score']})
- 평가: {assessment['overall']['message']}
- 경고 신호: {sum(1 for a in assessment['assessments'].values() if a['score'] <= 1)}개

### 개별 지표 상태:
- RP: {assessment['assessments']['RP']['level']} - {assessment['assessments']['RP']['message']}
- RRP: {assessment['assessments']['RRP']['level']} - {assessment['assessments']['RRP']['message']}
- 지준금: {assessment['assessments']['Reserves']['level']} - {assessment['assessments']['Reserves']['message']}
- 스프레드: {assessment['assessments']['Spread']['level']} - {assessment['assessments']['Spread']['message']}

## 딥다이브 분석 요청:

### 1. 유동성 환경 심층 분석 (7-10문장)
- 연준의 정책 사이클상 현재 위치 (QE/QT/긴축/완화)
- RP/RRP/지준금 3대 지표의 상호작용 분석
- 글로벌 유동성 흐름 맥락
- 금융시장 스트레스 수준 평가

### 2. 지표별 리스크 매트릭스 (상세 분석)
**RP (Repo) 분석:**
- 현재 수준의 역사적 위치
- 30/90일 변화율의 의미
- 은행 시스템 스트레스 평가

**RRP (Reverse Repo) 분석:**
- 시장 유동성 고갈 정도
- MMF 행동 패턴 분석
- 유동성 프리미엄 변화

**은행 지준금 분석:**
- 은행 대출 여력 평가
- QT 영향 분석
- 시스템 리스크 수준

**스프레드 분석:**
- 자금 조달 비용 압박
- 시장 기능 이상 여부
- 연준 개입 필요성

### 3. 다중 시나리오 분석 (각 확률 포함)
**Bull Case (낙관적 시나리오 __%):**
- 전개 조건 및 트리거
- 각 지표 예상 경로
- 자산 시장 반응

**Base Case (중립적 시나리오 __%):**
- 전개 조건
- 예상 지표 레인지
- 연준 정책 대응 시나리오

**Bear Case (비관적 시나리오 __%):**
- 전개 조건 및 위험 요인
- 유동성 위기 가능성
- 시장 충격 시나리오

### 4. 역사적 패턴 비교
- **2008 금융위기**: 유사점과 차이점
- **2020 코로나 위기**: 연준 대응 비교
- **2022-2023 긴축**: QT 국면 패턴
- **현재와의 차이점 및 시사점**

### 5. 자산군별 전략 (구체적 비중)
**주식:**
- 성장주 vs 가치주
- 섹터별 선호도

**채권:**
- 단기채 vs 장기채
- 크레딧 스프레드 전략

**대안자산:**
- 금/원자재
- 부동산/리츠
- 암호화폐

**현금 관리:**
- 최적 현금 비중
- MMF vs 단기채

### 6. 리스크 관리 프레임워크
**단기 리스크 (1-3개월):**
- 주요 모니터링 지표
- 즉시 대응 트리거

**중기 리스크 (3-12개월):**
- 구조적 변화 포인트
- 포지션 조정 타이밍

**장기 리스크 (12개월+):**
- 시스템적 리스크
- 전략적 자산배분

### 7. 모니터링 체크리스트
**일일 체크:**
- [ ] 주요 체크 지표 3가지

**주간 체크:**
- [ ] 주요 체크 지표 3가지

**월간 체크:**
- [ ] 주요 체크 지표 3가지

### 8. 트리거 레벨 (포지션 변경 조건)
- RP가 __B 초과 시 → 액션
- RRP가 __B 미만 시 → 액션
- 지준금이 __B 미만 시 → 액션
- 스프레드가 __bps 초과 시 → 액션

**전문가 수준으로, 하지만 실행 가능하게 작성해주세요. 수치와 근거를 명확히 제시하세요.**
"""
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        response = model.generate_content(
            prompt, 
            generation_config={
                'max_output_tokens': 4096,
                'temperature': 0.7
            },
            safety_settings=safety_settings
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            return "⚠️ AI 응답이 안전 필터에 의해 차단되었습니다. 다시 시도하세요."
        
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return "⚠️ API 할당량 초과. 잠시 후 다시 시도하세요."
        return f"⚠️ AI Deep Dive 분석 생성 중 오류: {str(e)}"

# ============================================================
# 5-2. ✨ Enhanced Gemini AI 분석 함수 (개별 지표 분석 - Deep Dive 강화)
# ============================================================
def generate_gemini_single_indicator(df, assessment, indicator, depth="기본"):
    """Gemini 2.0 Flash를 사용한 개별 지표 AI 분석 (Deep Dive 강화)"""
    
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
    last_90d = df.tail(90) if len(df) >= 90 else df
    
    # 변화율 계산
    if len(df) >= 7 and df[col].iloc[-7] != 0:
        change_7d = ((df[col].iloc[-1] - df[col].iloc[-7]) / df[col].iloc[-7]) * 100
    else:
        change_7d = 0.0
    
    if len(last_30d) >= 2 and last_30d[col].iloc[0] != 0:
        change_30d = ((last_30d[col].iloc[-1] - last_30d[col].iloc[0]) / last_30d[col].iloc[0]) * 100
    else:
        change_30d = 0.0
    
    if len(last_90d) >= 2 and last_90d[col].iloc[0] != 0:
        change_90d = ((last_90d[col].iloc[-1] - last_90d[col].iloc[0]) / last_90d[col].iloc[0]) * 100
    else:
        change_90d = 0.0
    
    ma7 = df[f"{col}_MA7"].iloc[-1]
    ma30 = df[f"{col}_MA30"].iloc[-1]
    ma60 = df[f"{col}_MA60"].iloc[-1]
    
    # 통계 지표
    std_30d = df[col].tail(30).std() if len(df) >= 30 else 0
    max_90d = df[col].tail(90).max() if len(df) >= 90 else df[col].max()
    min_90d = df[col].tail(90).min() if len(df) >= 90 else df[col].min()
    
    status_info = assessment["assessments"][key_for_assessment]
    
    # 깊이별 프롬프트 구성
    base_prompt = f"""
당신은 연준 유동성 지표 전문가입니다.
다음 하나의 지표에 대해서만 깊이 있게 분석해 주세요. 한국어로 답변해 주세요.

## 📊 분석 지표 정보
- **지표 이름**: {display_name}
- **최신 값**: {latest[col]:.2f}{unit}
- **변화율**: 7일 {change_7d:+.1f}% | 30일 {change_30d:+.1f}% | 90일 {change_90d:+.1f}%
- **이동평균**: MA7={ma7:.2f}{unit} | MA30={ma30:.2f}{unit} | MA60={ma60:.2f}{unit}
- **변동성**: 30일 표준편차 = {std_30d:.2f}{unit}
- **90일 범위**: 최고 {max_90d:.2f}{unit} ~ 최저 {min_90d:.2f}{unit}
- **현재 상태**: {status_info['level']} - {status_info['message']}
- **전체 유동성 종합**: {assessment['overall']['status']} (점수 {assessment['overall']['score']}/{assessment['overall']['max_score']})
"""

    # 깊이별 분석 요청사항
    if depth == "요약":
        analysis_request = """
## 📋 분석 요청 (요약 모드)
다음 항목을 **각 1~2문장**으로 간결하게 요약해 주세요:

1. 현재 수준 한 줄 요약
2. 단기 추세 (MA7 vs 현재값)
3. 주요 리스크 요인 1가지
4. 핵심 권고사항 1가지

**응답 형식**: 간결한 문장형, 불릿 포인트 최소 사용
"""
        max_tokens = 512
        
    elif depth == "기본":
        analysis_request = """
## 📋 분석 요청 (기본 모드)
다음 항목을 **각 2~3문장**으로 설명해 주세요:

1. 현재 수준과 최근 1~3개월 추세 요약
2. 이동평균(MA7/30/60) 관점에서 본 단기 vs 중기 추세
3. 경고/위험 레벨과의 거리 및 스트레스 정도 평가
4. 과거 유사 수준에서 나타났던 전형적인 시장 패턴
5. 투자자 관점에서의 리스크 요인과 잠재적 기회
6. 앞으로 주시해야 할 트리거 레벨과 대응 전략

**응답 형식**: 문단형 위주, 필요시 bullet 3~5개 이내
"""
        max_tokens = 1024
        
    else:  # 딥다이브
        analysis_request = f"""
## 📋 분석 요청 (딥다이브 모드)
다음 항목을 **매우 상세하게** 분석해 주세요:

### 🔍 기본 분석 (상세)
1. **현재 수준 정밀 평가**
   - 절대값 수준 평가 (역사적 백분위수)
   - 30일/90일 변화율의 의미
   - 현재값이 MA7/30/60 대비 어느 위치인지 구체적 설명

2. **이동평균 크로스오버 분석**
   - MA7-MA30 골든크로스/데드크로스 여부
   - MA30-MA60 중기 트렌드
   - 이동평균 수렴/발산 패턴의 의미

3. **경고 레벨 분석**
   - 현재 정상/경고/위험 구간 위치
   - 각 임계값까지의 거리 ({unit} 단위)
   - 현재 변화율로 임계값 도달까지 예상 기간

4. **변동성 분석**
   - 30일 표준편차 {std_30d:.2f}{unit}의 의미
   - 최근 변동성이 과거 대비 높은지/낮은지
   - 변동성 급증/급감 시그널 여부

### 📚 역사적 패턴 분석
5. **2008 금융위기 패턴과의 비교**
   - 유사점과 차이점
   - 당시 이 지표 수준과 현재 비교

6. **2020 코로나 위기 패턴과의 비교**
   - 유동성 급변 시기와의 유사성
   - 연준 대응과 시장 반응 패턴

7. **2022-2023 긴축 사이클과의 비교**
   - QT(양적긴축) 국면에서의 패턴
   - 현재와의 차이점

### 💡 투자 전략 (구체적)
8. **자산군별 영향 분석**
   - 주식: 성장주 vs 가치주
   - 채권: 단기채 vs 장기채
   - 금/원자재
   - 암호화폐 (비트코인)

9. **리스크 시나리오 (정량적)**
   - **Best Case (30% 확률)**: 어떤 수준? 투자 전략?
   - **Base Case (50% 확률)**: 어떤 수준? 투자 전략?
   - **Worst Case (20% 확률)**: 어떤 수준? 투자 전략?

10. **모니터링 체크리스트**
    - 주간 체크: 어떤 수치 변화 주시?
    - 월간 체크: 어떤 추세 변화 주시?
    - 즉시 알람: 어떤 임계값 돌파 시 긴급 대응?

**응답 형식**: 
- 각 섹션 헤더 명확히 구분
- 구체적 수치와 함께 설명
- bullet points 적극 활용 (각 항목 5~10개)
- 표나 리스트 형태로 정리된 정보 포함
"""
        max_tokens = 3072
    
    final_prompt = base_prompt + analysis_request
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        generation_config = {
            'max_output_tokens': max_tokens,
            'temperature': 0.7,
        }
        response = model.generate_content(final_prompt, generation_config=generation_config)
        
        # 응답에 메타 정보 추가
        meta_info = f"""
---
**📌 분석 메타 정보**
- 분석 모드: {depth}
- 분석 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 토큰 한도: {max_tokens}
- AI 모델: Gemini 2.0 Flash Experimental

---

"""
        return meta_info + response.text
        
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return f"""
### ⚠️ Gemini API 할당량 초과

현재 Gemini API의 무료 tier 할당량을 초과했습니다.

**해결 방법:**
1. **잠시 대기**: 10-60분 후 다시 시도
2. **할당량 확인**: [사용량 확인](https://ai.dev/usage?tab=rate-limit)

**현재 {display_name} 상태 요약 (수동 분석):**
- 현재값: {latest[col]:.2f}{unit}
- 7일 변화: {change_7d:+.1f}%
- 30일 변화: {change_30d:+.1f}%
- 상태: {status_info['level']}
- 평가: {status_info['message']}
"""
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
            # ============ 수정된 부분 시작 ============
            st.markdown("#### 종합 유동성 분석")
            
            col_depth, col_btn = st.columns([3, 1])
            
            with col_depth:
                comprehensive_depth = st.select_slider(
                    "분석 깊이", 
                    ["요약", "기본", "딥다이브"], 
                    value="기본",
                    help="요약: 간결한 핵심 분석 / 기본: 표준 분석 / 딥다이브: 상세한 심층 분석"
                )
            
            with col_btn:
                st.write("")
                st.write("")
                run_comprehensive = st.button("🚀 종합 AI 분석 실행", type="primary", key="comprehensive_analysis_btn")
            
            if run_comprehensive:
                with st.spinner(f"🧠 Gemini {'심층' if comprehensive_depth == '딥다이브' else ''} 분석 중..."):
                    try:
                        # 분석 깊이에 따라 다른 함수 호출
                        if comprehensive_depth == "딥다이브":
                            ai_analysis = generate_gemini_analysis_deep_dive(df, assessment)
                        else:
                            ai_analysis = generate_gemini_analysis(df, assessment, depth=comprehensive_depth)
                        
                        st.session_state['comprehensive_analysis'] = ai_analysis
                        st.session_state['comprehensive_depth'] = comprehensive_depth
                    except Exception as e:
                        st.error(f"분석 중 오류: {str(e)}")
            # ============ 수정된 부분 끝 ============
            
            if 'comprehensive_analysis' in st.session_state:
                # 분석 깊이 표시
                depth_badge = st.session_state.get('comprehensive_depth', '기본')
                depth_colors = {
                    "요약": "#4CAF50",
                    "기본": "#2196F3", 
                    "딥다이브": "#FF6B35"
                }
                
                st.markdown(
                    f"""
                    <div style='padding: 10px; border-radius: 5px; background-color: {depth_colors.get(depth_badge, '#2196F3')}20; 
                         border-left: 4px solid {depth_colors.get(depth_badge, '#2196F3')}; margin-bottom: 20px;'>
                        <strong>📊 분석 모드:</strong> {depth_badge}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown(st.session_state['comprehensive_analysis'])
                
                st.download_button(
                    "📥 종합 분석 다운로드",
                    st.session_state['comprehensive_analysis'],
                    f"comprehensive_analysis_{depth_badge}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                    "text/markdown"
                )
        
        else:  # 개별 지표 분석 (Deep Dive 강화)
            st.markdown("#### 🔬 개별 지표 Deep Dive 분석")
            
            # 2단 레이아웃
            col_ind, col_depth = st.columns([1, 1])
            
            with col_ind:
                indicator = st.selectbox(
                    "📊 분석할 지표 선택", 
                    ["RP", "RRP", "Reserves", "Spread"],
                    help="심층 분석할 단일 지표를 선택하세요"
                )
            
            with col_depth:
                # 깊이별 설명
                depth_info = {
                    "⚡ 요약": {
                        "time": "~1분",
                        "desc": "핵심만 빠르게",
                        "tokens": "512",
                        "color": "#90EE90"
                    },
                    "📊 기본": {
                        "time": "~3분",
                        "desc": "균형잡힌 분석",
                        "tokens": "1,024",
                        "color": "#87CEEB"
                    },
                    "🔬 딥다이브": {
                        "time": "~5분",
                        "desc": "매우 상세한 분석",
                        "tokens": "3,072",
                        "color": "#FFB6C1"
                    }
                }
                
                depth_display = st.select_slider(
                    "🎚️ 분석 깊이 선택",
                    options=list(depth_info.keys()),
                    value="📊 기본",
                    help="슬라이더를 움직여 원하는 분석 깊이를 선택하세요"
                )
                
                # 선택된 깊이 정보 표시
                selected_info = depth_info[depth_display]
                st.info(
                    f"**예상 분석 시간**: {selected_info['time']} | "
                    f"**토큰**: {selected_info['tokens']} | "
                    f"**특징**: {selected_info['desc']}"
                )
            
            # depth 값 추출 (이모지 제거)
            depth = depth_display.split()[1]  # "요약", "기본", "딥다이브"
            
            # 실행 버튼
            col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
            
            with col_btn1:
                run_analysis = st.button(
                    f"🚀 {depth} 분석 실행", 
                    type="primary",
                    use_container_width=True,
                    key="indicator_analysis_btn"
                )
            
            with col_btn2:
                if 'single_analysis' in st.session_state:
                    clear_analysis = st.button(
                        "🗑️ 초기화",
                        use_container_width=True,
                        key="clear_analysis_btn"
                    )
                    if clear_analysis:
                        del st.session_state['single_analysis']
                        del st.session_state['single_indicator']
                        del st.session_state['single_depth']
                        st.rerun()
            
            # 분석 실행
            if run_analysis:
                with st.spinner(f"🧠 Gemini가 {indicator} 지표를 {depth} 분석 중..."):
                    ai_single = generate_gemini_single_indicator(df, assessment, indicator, depth)
                    st.session_state['single_analysis'] = ai_single
                    st.session_state['single_indicator'] = indicator
                    st.session_state['single_depth'] = depth
            
            # 결과 표시
            if 'single_analysis' in st.session_state:
                st.markdown("---")
                
                # 헤더 정보
                indicator_name = {
                    "RP": "RP (Repo)",
                    "RRP": "RRP (Reverse Repo)",
                    "Reserves": "은행 지준금",
                    "Spread": "SOFR-IORB 스프레드"
                }.get(st.session_state.get('single_indicator', ''), '')
                
                st.markdown(
                    f"### 📊 {indicator_name} - {st.session_state.get('single_depth', '')} 분석 결과"
                )
                
                # 분석 내용
                st.markdown(st.session_state['single_analysis'])
                
                # 다운로드 및 공유 옵션
                col_dl1, col_dl2 = st.columns([1, 1])
                
                with col_dl1:
                    st.download_button(
                        "📥 분석 결과 다운로드 (Markdown)",
                        st.session_state['single_analysis'],
                        f"{st.session_state.get('single_indicator', 'indicator')}_"
                        f"{st.session_state.get('single_depth', 'analysis')}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                        "text/markdown",
                        use_container_width=True,
                        key="download_md_btn"
                    )
                
                with col_dl2:
                    # TXT 형식으로도 제공
                    st.download_button(
                        "📄 텍스트 형식 다운로드",
                        st.session_state['single_analysis'],
                        f"{st.session_state.get('single_indicator', 'indicator')}_"
                        f"{st.session_state.get('single_depth', 'analysis')}_"
                        f"{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        "text/plain",
                        use_container_width=True,
                        key="download_txt_btn"
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
            if st.button("🧹 대화 지우기", use_container_width=True, key="clear_chat_btn"):
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
        rrp_change_7d = ((df['RRP'].iloc[-1] - df['RRP'].iloc[-7]) / df['RRP'].iloc[-7]) * 100 if df['RRP'].


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
        "text/csv",
        key="download_csv_btn"
    )

with col2_d:
    report = f"""

st.download_button(
        "📄 종합 리포트 다운로드 (TXT)",
        report,
        f"liquidity_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
        "text/plain",
        key="download_report_btn"
    )

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🏦 연준 유동성 스트레스 모니터링 대시보드 v3.0</p>
        <p>데이터 출처: FRED (Federal Reserve Economic Data) | AI: Gemini 2.0 Flash + OpenAI GPT-4</p>
        <p>⚠️ 본 분석은 투자 권유가 아니며, 참고 목적으로만 활용하시기 바랍니다.</p>
    </div>
    """,
    unsafe_allow_html=True
)




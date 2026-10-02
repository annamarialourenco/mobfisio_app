import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# Limiares operacionais calibrados
limiar_youden = 0.38  # 38,0% - Ponto de corte do modelo (Youden)
limiar_alto_risco = 0.50  # 50,0% - Alerta crítico de suspensão

# 1. Configuração da Página
st.set_page_config(
    page_title="SSDC MobFisio IA",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Gestão de Estado (Valores Padrão e Reset)
DEFAULTS = {
    "pam_input": 75,
    "fc_input": 85,
    "fr_input": 18,
    "spo2_input": 96,
    "dva_input": 0,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

if "foi_calculado" not in st.session_state:
    st.session_state["foi_calculado"] = False


def resetar_campos():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state["foi_calculado"] = False


# 3. CSS Personalizado (Estilização Hospitalar em Modo Escuro)
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; }
    
    /* Cabeçalho */
    .header-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .header-title {
        color: #F8FAFC;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle { color: #94A3B8; font-size: 0.9rem; margin-top: 4px; }
    .header-badge {
        background-color: #0284C7;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
    }

    /* Cards de Resultado Customizados */
    .result-card { 
        border-radius: 12px; 
        padding: 18px; 
        margin-top: 15px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .card-red { background: rgba(127, 29, 29, 0.3); border: 1.5px solid #EF4444; }
    .card-yellow { background: rgba(120, 53, 15, 0.3); border: 1.5px solid #F59E0B; }
    .card-green { background: rgba(6, 78, 59, 0.3); border: 1.5px solid #10B981; }

    .status-badge-red { background-color: #EF4444; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .status-badge-yellow { background-color: #F59E0B; color: black; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }
    .status-badge-green { background-color: #10B981; color: white; padding: 4px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem; }

    /* Métrica Hero */
    .hero-metric-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .hero-metric-title { color: #94A3B8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .hero-metric-value { font-size: 2.8rem; font-weight: 800; margin: 4px 0; }
    .hero-metric-sub { color: #64748B; font-size: 0.8rem; }
    
    /* Estado Inicial / Aguardando */
    .placeholder-card {
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 35px 20px;
        text-align: center;
        color: #94A3B8;
    }

    /* Rodapé / Disclaimer */
    .footer-disclaimer {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 25px;
        text-align: center;
        color: #64748B;
        font-size: 0.78rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# 4. Função para Carregar o Modelo e Metadados do Dicionário
@st.cache_resource
def carregar_modelo():
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(
        pasta_atual, "ssdc_mobfisio_xgboost_v1.joblib"
    )

    dados_carregados = joblib.load(caminho_modelo)

    if isinstance(dados_carregados, dict):
        modelo = dados_carregados.get("modelo") or dados_carregados.get("model")
        limiar = dados_carregados.get("limiar_calibrado", 0.38)
        features = dados_carregados.get(
            "features",
            ["pam_pre", "fc_pre", "fr_pre", "spo2_pre", "dva_em_uso_pre"],
        )
    else:
        modelo = dados_carregados
        limiar = 0.38
        features = ["pam_pre", "fc_pre", "fr_pre", "spo2_pre", "dva_em_uso_pre"]

    return modelo, limiar, features


modelo_ssdc, limiar_calibrado_salvo, features_treino = carregar_modelo()

# 5. Cabeçalho Principal
st.markdown(
    """
    <div class="header-card">
        <div class="header-title">
            🏥 SSDC MobFisio IA 
            <span class="header-badge">UTI Adulto v1.0</span>
        </div>
        <div class="header-subtitle">
            Sistema de Suporte à Decisão Clínica para Estratificação de Risco na Mobilização Precoce
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 6. Formulário de Entrada dos Dados (Barra Lateral)
st.sidebar.header("📋 Parâmetros Pré-Mobilização")
st.sidebar.caption("Insira os sinais vitais coletados à beira do leito ($t_{0}$):")

with st.sidebar.form(key="form_parametros_vitais"):
    pam_input = st.number_input(
        "Pressão Arterial Média — PAM (mmHg)",
        min_value=30,
        max_value=160,
        step=1,
        key="pam_input",
        help="Principal determinante de risco (Peso de 60,7% no modelo).",
    )

    fc_input = st.number_input(
        "Frequência Cardíaca — FC (bpm)",
        min_value=30,
        max_value=200,
        step=1,
        key="fc_input",
    )

    fr_input = st.number_input(
        "Frequência Respiratória — FR (ipm)",
        min_value=8,
        max_value=50,
        step=1,
        key="fr_input",
    )

    spo2_input = st.number_input(
        "Saturação de Oxigênio — SpO₂ (%)",
        min_value=50,
        max_value=100,
        step=1,
        key="spo2_input",
    )

    dva_input = st.radio(
        "Droga Vasoativa (DVA) em Uso?",
        options=[0, 1],
        format_func=lambda x: "Não (0)" if x == 0 else "Sim (1)",
        key="dva_input",
    )

    # Botão de Ação do Formulário
    btn_calcular = st.form_submit_button(
        "🔍 Calcular Risco", use_container_width=True, type="primary"
    )

# Botão para Resetar Parâmetros
st.sidebar.button(
    "🔄 Limpar / Restaurar Padrão",
    on_click=resetar_campos,
    use_container_width=True,
)

if btn_calcular:
    st.session_state["foi_calculado"] = True

# 7. Painel Principal de Resultados
col_esq, col_dir = st.columns([1.1, 1.3])

if st.session_state["foi_calculado"]:
    status_pam = "⚠️" if (pam_input < 65 or pam_input > 110) else "✅"
    status_fc = "⚠️" if (fc_input < 60 or fc_input > 100) else "✅"
    status_fr = "⚠️" if (fr_input < 12 or fr_input > 20) else "✅"
    status_spo2 = "⚠️" if spo2_input < 92 else "✅"
    status_dva = "⚠️" if dva_input == 1 else "✅"

    with col_esq:
        st.subheader("📊 Sinais Vitais do Paciente ($t_0$)")

        df_vitais = pd.DataFrame(
            {
                "Status": [status_pam, status_fc, status_fr, status_spo2, status_dva],
                "Parâmetro": ["PAM", "FC", "FR", "SpO₂", "Suporte DVA"],
                "Valor": [
                    f"{pam_input} mmHg",
                    f"{fc_input} bpm",
                    f"{fr_input} ipm",
                    f"{spo2_input}%",
                    "Em Uso" if dva_input == 1 else "Isento",
                ],
                "Referência": [
                    "65 - 110 mmHg",
                    "60 - 100 bpm",
                    "12 - 20 ipm",
                    "≥ 92%",
                    "Sem DVA",
                ],
            }
        )
        st.dataframe(df_vitais, use_container_width=True, hide_index=True)

    with col_dir:
        st.subheader("🛡 Parecer Preditivo de Segurança")

        if modelo_ssdc is not None:
            # Cria o DataFrame garantindo a ordem exata das colunas do treinamento
            dados_paciente = pd.DataFrame(
                [{
                    "pam_pre": float(pam_input),
                    "fc_pre": float(fc_input),
                    "fr_pre": float(fr_input),
                    "spo2_pre": float(spo2_input),
                    "dva_em_uso_pre": float(dva_input),
                }]
            )[features_treino]

            # Predição da probabilidade
            if hasattr(modelo_ssdc, "predict_proba"):
                prob_array = modelo_ssdc.predict_proba(dados_paciente)
                probabilidade = float(prob_array[0][1])
            else:
                booster = (
                    modelo_ssdc.get_booster()
                    if hasattr(modelo_ssdc, "get_booster")
                    else modelo_ssdc
                )
                dmatrix = xgb.DMatrix(dados_paciente)
                pred_raw = booster.predict(dmatrix)
                probabilidade = (
                    float(pred_raw[0][1]) if pred_raw.ndim == 2 else float(pred_raw[0])
                )

            prob_percentual = probabilidade * 100

            # Lógica de Cores e Classificação em 3 Zonas
            if probabilidade < limiar_youden:
                cor_metric = "#10B981"  # Verde
            elif limiar_youden <= probabilidade < limiar_alto_risco:
                cor_metric = "#F59E0B"  # Amarelo
            else:
                cor_metric = "#EF4444"  # Vermelho

            # Hero Metric Card
            st.markdown(
                f"""
                <div class="hero-metric-container">
                    <div class="hero-metric-title">Probabilidade Estimada de Instabilidade</div>
                    <div class="hero-metric-value" style="color: {cor_metric};">{prob_percentual:.1f}%</div>
                    <div class="hero-metric-sub">Ponto de Corte Youden (ML): 38,0%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(probabilidade, 1.0))

            # Card de Conduta Clínica
            if probabilidade < limiar_youden:
                st.markdown(
                    """
                    <div class="result-card card-green">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.05rem; font-weight: 700; color: #A7F3D0;">✅ LIBERAÇÃO CONFIRMADA</span>
                            <span class="status-badge-green">NÍVEL VERDE</span>
                        </div>
                        <p style="color: #F8FAFC; margin-bottom: 6px;"><b>Paciente em Faixa Segura de Estabilidade.</b></p>
                        <p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;"><b>Recomendação:</b> Proceder com o protocolo de mobilização precoce habitual sob monitorização padrão à beira do leito.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif limiar_youden <= probabilidade < limiar_alto_risco:
                st.markdown(
                    """
                    <div class="result-card card-yellow">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.05rem; font-weight: 700; color: #FDE68A;">⚠️ ATENÇÃO REQUERIDA</span>
                            <span class="status-badge-yellow">NÍVEL AMARELO</span>
                        </div>
                        <p style="color: #F8FAFC; margin-bottom: 6px;"><b>Risco Moderado de Instabilidade (Faixa de Vigilância).</b></p>
                        <p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;"><b>Recomendação:</b> Proceder com mobilização adaptada/passiva sob monitorização contínua de SpO₂ e PAM durante a conduta.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="result-card card-red">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 1.05rem; font-weight: 700; color: #FCA5A5;">🚨 ALERTA DE SEGURANÇA</span>
                            <span class="status-badge-red">NÍVEL VERMELHO</span>
                        </div>
                        <p style="color: #F8FAFC; margin-bottom: 6px;"><b>Risco Elevado de Instabilidade Hemodinâmica ou Respiratória.</b></p>
                        <p style="color: #CBD5E1; font-size: 0.9rem; margin: 0;"><b>Recomendação:</b> Suspender ou adiar a mobilização no momento ($t_0$), reavaliar parâmetros hemodinâmicos e alinhar conduta com a equipe multiprofissional.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
else:
    # Estado Inicial
    with col_esq:
        st.info("👈 Insira os parâmetros na barra lateral e clique em **Calcular Risco**.")

    with col_dir:
        st.markdown(
            """
            <div class="placeholder-card">
                <h3>🩺 Aguardando Análise</h3>
                <p>A avaliação do modelo XGBoost será exibida aqui após submeter os sinais vitais.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

with st.expander("ℹ️ Entenda a Lógica de Decisão do SSDC MobFisio IA"):
    st.write(
        """
        - **Estratificação de Risco por Faixas Operacionais:**
          - 🟩 **Risco Baixo (< 38,0%):** Probabilidade abaixo do limiar calibrado. Liberação padrão para o protocolo de mobilização precoce.
          - 🟨 **Risco Moderado / Atenção (38,0% - 49,9%):** Faixa de vigilância correspondente ao risco basal de mobilização em UTI. Autorizado mediante monitorização rigorosa e conduta adaptada.
          - 🟥 **Risco Crítico (≥ 50,0%):** Probabilidade expressiva de intercorrência. Recomendação de adiar ou reavaliar antes da conduta ($t_0$).
        - **Peso dos Sinais Vitais:** A Pressão Arterial Média (PAM) representa **60,7%** do peso na árvore do XGBoost, seguida pelo uso de DVA e Frequência Respiratória.
        """
    )

# 8. Rodapé de Responsabilidade Profissional e Isenção de Responsabilidade Legal
st.markdown(
    """
    <div class="footer-disclaimer">
        <b>⚖️ AVISO DE RESPONSABILIDADE CLÍNICA E LEGAL:</b><br>
        O <b>SSDC MobFisio IA</b> é uma ferramenta de suporte preditivo desenvolvida exclusivamente para auxílio na tomada de decisão. 
        As estimativas fornecidas <b>não substituem a avaliação clínica individualizada</b>, a anamnese detalhada e o julgamento técnico do fisioterapeuta assistente ou da equipe multiprofissional.<br>
        A decisão final quanto ao protocolo de mobilização precoce é de <b>inteira e exclusiva responsabilidade do profissional de saúde responsável</b> à beira do leito.
    </div>
    """,
    unsafe_allow_html=True,
)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image
import matplotlib as mpl
mpl.use("agg")

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="380" height="390" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)




# -- Figure type list
fig_type_list = ['By different Ni', 'By different IC', 'By different parameters']
ab_match_list = ['KD', 't0', 'SR']
Ni_list = ['min', 'med', 'max']

# Title the app
st.title('PPS of KD curved universe')

st.markdown("""
 * Use the menu at left to select figure type and set plot parameters
 * Your plots will appear below
""")


#-- Select what to compare (different Ni or IC)
select_event = st.sidebar.selectbox('What you want to compare with in your figure?', fig_type_list)

if select_event == fig_type_list[0]:
    st.markdown("""
    * The Ni [min, med, max] is: [11.713, 12.7, 13.6]
    * Smaller Ni corresponds to larger Omega_K
    """)
    #-- Choose where to match a&b. In KD, start of inf,or SR
    ab_match = st.sidebar.selectbox('Where you want to match a&b?', ab_match_list)
    
    st.subheader("log(a) and log(b) evolution:")
    image = Image.open('figure_streamlit/Diff_Ni/ab_evol/ab_evol-diffNi-ab_' + ab_match + '.png')
    st.image(image, caption='ab_evolution')

    st.subheader("Primordial power spectrum:")
    pdf_path = 'figure_streamlit/Diff_Ni/PPS/PPS-diffNi-ab_' + ab_match + '.pdf'
    displayPDF(pdf_path)      

    st.subheader("Cosmic microwave background in low l:")
    pdf_path = 'figure_streamlit/Diff_Ni/CMB/CMB-diffNi-ab_' + ab_match + '.pdf'
    displayPDF(pdf_path)

elif select_event == fig_type_list[1]:

    #-- Choose Ni
    Ni = st.sidebar.selectbox('Set Ni', Ni_list)
    
    ## Match a&b at different place:
    st.header('Match a&b at different place:')

    st.subheader("log(a) and log(b) evolution:")
    image = Image.open('figure_streamlit/Diff_IC/diff_ab_match/ab_evol/ab_evol-diffIC-Ni_'+ Ni +'.png')
    st.image(image, caption='ab_evolution')

    st.subheader("Primordial power spectrum")
    pdf_path = 'figure_streamlit/Diff_IC/diff_ab_match/PPS/PPS-diffIC-abmatch-Ni_'+ Ni +'.pdf'
    displayPDF(pdf_path)

    st.subheader("Cosmic microwave background in low l:")
    pdf_path = 'figure_streamlit/Diff_IC/diff_ab_match/CMB/CMB-diffIC-abmatch-Ni_'+ Ni +'.pdf'
    displayPDF(pdf_path)

    ## Other ICs
    st.header('Choose other ICs (BD, flat-RSET etc):')
    st.subheader("Primordial power spectrum")
    image = Image.open('figure_streamlit/Diff_IC/diff_IC_method/PPS/PPS-diffIC-Ni_'+ Ni +'.png')
    st.image(image, caption='PPS')

    st.subheader("Cosmic microwave background in low l:")
    pdf_path = 'figure_streamlit/Diff_IC/diff_IC_method/CMB/CMB-diffIC-Ni_'+ Ni +'.pdf'
    displayPDF(pdf_path)

elif select_event == fig_type_list[2]:
    st.markdown("""
 * The is evolution of Omega_K: 1/(aH)^2
 * With varying parameters: ns, As, omega_K, omega_m
 * Original parameter set: N_i=0.5, V=Starobinsky, H0=64.03, ns=0.96535, logA=3.0336, Omega_K=-0.0092, Omega_m=0.3453
""")

    st.subheader("ns:")
    image = Image.open('figure_streamlit/Omega_K/OmegaK-ns-Ni0_5.png')

    st.subheader("As:")
    image = Image.open('figure_streamlit/Omega_K/OmegaK-As-Ni0_5.png')

    st.subheader("omega_K:")
    image = Image.open('figure_streamlit/Omega_K/OmegaK-OmegaK-Ni0_5.png')

    st.subheader("omega_m:")
    image = Image.open('figure_streamlit/Omega_K/OmegaK-Omegam-Ni0_5.png')
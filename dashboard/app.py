import streamlit as st


# --------------------------------------------------
# Streamlit Navigation
# --------------------------------------------------

pages = [
    st.Page(
        "pages/Regression.py",
        title="Regression",
        icon="📈"
    ),
    st.Page(
        "pages/Classification.py",
        title="Classification",
        icon="🎯"
    ),
   ]


pg = st.navigation(pages)

pg.run()
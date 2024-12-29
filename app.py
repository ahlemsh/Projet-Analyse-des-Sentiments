import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password123")

# Charger les modèles et vectorizers
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Charger ou créer l'historique
try:
    historique = pd.read_csv("historique_avis.csv")
except FileNotFoundError:
    historique = pd.DataFrame(columns=["ID Utilisateur", "Commentaire", "Sentiment"])

# Fonction pour sauvegarder l'historique
def sauvegarder_historique(historique):
    historique.to_csv("historique_avis.csv", index=False)

# Ajouter du CSS pour le style
st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f4f4f4;
        }
        h1 {
            color: #4CAF50;
            text-align: center;
        }
        .stButton>button {
            background-color: transparent;
            border: none;
            padding: 0;
            font-size: 10px;
            cursor: pointer;
            display: inline;
            line-height: normal;
        }
        .stButton>button:hover {
            color: #ff0000;
        }
        .table-header {
            font-weight: bold;
            background-color: #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title("Analyse d'Avis Client 🌟")

# Sidebar pour le rôle
option = st.sidebar.radio("Sélectionnez votre profil :", ["Client", "Admin"])

if option == "Client":
    st.header("Soumettez votre avis 📝")
    user_id = st.text_input("Entrez votre ID :")
    user_review = st.text_area("Écrivez votre avis ici :")
    
    if st.button("Soumettre"):
        if user_id and user_review:
            try:
                review_vectorized = vectorizer.transform([user_review])
                sentiment = model.predict(review_vectorized)[0]
                sentiment_label = "Positif" if sentiment == 1 else "Négatif" if sentiment == -1 else "Neutre"
                
                nouveau_commentaire = {"ID Utilisateur": user_id, "Commentaire": user_review, "Sentiment": sentiment_label}
                historique = pd.concat([historique, pd.DataFrame([nouveau_commentaire])], ignore_index=True)
                sauvegarder_historique(historique)
                
                if sentiment == 1:
                    st.success("🌟 Avis positif : Merci pour votre retour positif ! 😊")
                elif sentiment == -1:
                    st.error("🚨 Avis négatif : Nous sommes désolés pour votre expérience 😔")
                else:
                    st.info("🤔 Avis neutre : Merci pour votre avis !")
            except Exception as e:
                st.error(f"Une erreur s'est produite : {e}")
        else:
            st.warning("⚠️ Veuillez remplir tous les champs.")

elif option == "Admin":
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.header("Authentification Admin")
        username = st.text_input("Nom d'utilisateur")
        password = st.text_input("Mot de passe", type="password")
        
        if st.button("Se connecter"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("Authentification réussie !")
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")
    else:
        st.header("Bienvenue dans l'espace Admin 🛠️")
        
        # Boutons pour l'interface Admin
        action = st.radio("Choisissez une action :", ["Tableau de bord", "Analyse des statistiques"])
        
        if action == "Tableau de bord":
            st.subheader("Tableau de bord des avis 📊")
            if not historique.empty:
                # Filtrage par type d'avis
                type_avis = st.selectbox("Filtrer par type d'avis", ["Tous", "Positif", "Négatif", "Neutre"])
                if type_avis != "Tous":
                    historique_filtre = historique[historique['Sentiment'] == type_avis]
                else:
                    historique_filtre = historique
                
                # Pagination avec une barre pour la navigation
                avis_par_page = 4
                total_pages = (len(historique_filtre) - 1) // avis_par_page + 1
                current_page = st.number_input("Page", min_value=1, max_value=total_pages, step=1) - 1
                debut = current_page * avis_par_page
                fin = debut + avis_par_page
                historique_page = historique_filtre.iloc[debut:fin]

                # Affichage du tableau avec suppression par icône uniquement
                st.write("### Avis")
                if not historique_page.empty:
                    col0, col1, col2, col3, col4 = st.columns([1, 2, 4, 2, 1])
                    col0.write("**#**")
                    col1.write("**User ID**")
                    col2.write("**Avis**")
                    col3.write("**Sentiment**")
                    col4.write("")
                    for i, (row_index, row) in enumerate(historique_page.iterrows(), start=debut + 1):
                        with col0:
                            st.write(i)
                        with col1:
                            st.write(row['ID Utilisateur'])
                        with col2:
                            st.write(row['Commentaire'])
                        with col3:
                            st.write(row['Sentiment'])
                        with col4:
                            if st.button("🗑️", key=f"delete_{row_index}"):
                                historique = historique.drop(row_index).reset_index(drop=True)
                                sauvegarder_historique(historique)

                st.write(f"Page {current_page + 1} sur {total_pages}")

                # Téléchargement du CSV
                csv = historique.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger l'historique en CSV",
                    data=csv,
                    file_name='historique_avis.csv',
                    mime='text/csv'
                )
            else:
                st.info("Aucun avis soumis pour l'instant.")
        
        elif action == "Analyse des statistiques":
            st.subheader("Statistiques et graphiques 📈")
            
            # Statistiques générales
            st.subheader("Statistiques générales")
            if not historique.empty:
                total_avis = len(historique)
                avis_positifs = len(historique[historique["Sentiment"] == "Positif"])
                avis_negatifs = len(historique[historique["Sentiment"] == "Négatif"])
                avis_neutres = len(historique[historique["Sentiment"] == "Neutre"])
                
                st.write(f"**Total des avis soumis :** {total_avis}")
                st.write(f"**Avis positifs :** {avis_positifs} ({(avis_positifs / total_avis) * 100:.2f}%)")
                st.write(f"**Avis négatifs :** {avis_negatifs} ({(avis_negatifs / total_avis) * 100:.2f}%)")
                st.write(f"**Avis neutres :** {avis_neutres} ({(avis_neutres / total_avis) * 100:.2f}%)")
                
                # Distribution des sentiments
                sentiments_counts = historique["Sentiment"].value_counts()
                
                # Diagramme circulaire des sentiments
                st.markdown("### Distribution des sentiments")
                fig, ax = plt.subplots()
                ax.pie(sentiments_counts, labels=sentiments_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Pour un cercle parfait
                st.pyplot(fig)
                
                # Histogramme des sentiments
                st.markdown("### Nombre d'avis par sentiment")
                fig, ax = plt.subplots()
                sentiments_counts.plot(kind="bar", color=["green", "red", "gray"], ax=ax)
                ax.set_ylabel("Nombre d'avis")
                ax.set_xlabel("Sentiment")
                ax.set_title("Histogramme des sentiments")
                st.pyplot(fig)
            else:
                st.info("Aucune donnée disponible pour les statistiques.")
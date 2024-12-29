import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

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
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
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
            if username == "admin" and password == "password123":  # Remplacez par des identifiants sécurisés
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
                st.dataframe(historique)
                
                # Supprimer un avis
                st.markdown("### Supprimer un avis")
                avis_index = st.number_input("Entrez l'index de l'avis à supprimer :", min_value=0, max_value=len(historique)-1, step=1)
                if st.button("Supprimer"):
                    historique = historique.drop(index=avis_index).reset_index(drop=True)
                    sauvegarder_historique(historique)
                    st.success("Avis supprimé avec succès !")
                
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
            if not historique.empty:
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
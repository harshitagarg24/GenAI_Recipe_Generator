import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="🍽 Smart Recipe Generator", layout="wide")

# ---------- STYLING ----------
st.markdown("""
    <style>
        body {background-color: #faf6ff;}
        .block-container {padding-top: 2rem;}
        h1 {text-align: center; color: #6a0dad; font-size: 3rem; font-weight: bold;}
        .recipe-card {
            border-radius: 15px;
            background: linear-gradient(135deg, #f8f4ff, #fff);
            box-shadow: 0px 4px 10px rgba(150, 100, 200, 0.2);
            padding: 15px;
            margin-bottom: 10px;
        }
        .stButton>button {
            background-color: #6a0dad !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ---------- DATA GENERATION ----------
def generate_recipes(cuisine, recipe_list):
    recipes = []
    for r in recipe_list:
        recipes.append({
            "cuisine": cuisine,
            "recipe": r["name"],
            "ingredients": ", ".join(r["ingredients"]),
            "instructions": r["instructions"],
            "image": r["image"],
            "difficulty": random.choice(["Easy", "Medium", "Hard"]),
            "rating": round(random.uniform(3.5, 5.0), 1)
        })
    return recipes

# ---------- SAMPLE RECIPE DATA ----------
recipe_data = {
    "Chinese": [
        {"name": "Veg Fried Rice", "ingredients": ["rice", "soy sauce", "vegetables", "garlic"], "instructions": "Stir fry rice and veggies with soy sauce.", "image": "https://www.themealdb.com/images/media/meals/1529444830.jpg"},
        {"name": "Chicken Manchurian", "ingredients": ["chicken", "corn flour", "soy sauce", "ginger", "garlic"], "instructions": "Deep fry chicken and cook in soy-garlic sauce.", "image": "https://www.themealdb.com/images/media/meals/1529446352.jpg"},
    ],
    "North Indian": [
        {"name": "Paneer Butter Masala", "ingredients": ["paneer", "tomato", "butter", "cream"], "instructions": "Cook paneer in butter tomato gravy.", "image": "https://www.themealdb.com/images/media/meals/uttuxy1511382180.jpg"},
        {"name": "Chole Bhature", "ingredients": ["chickpeas", "flour", "onion", "tomato"], "instructions": "Cook chickpeas curry and serve with fried bread.", "image": "https://www.themealdb.com/images/media/meals/sypxpx1515365095.jpg"},
    ],
    "South Indian": [
        {"name": "Masala Dosa", "ingredients": ["rice", "urad dal", "potato", "curry leaves"], "instructions": "Make dosa from batter and fill with spiced potato.", "image": "https://www.themealdb.com/images/media/meals/vytypy1511883765.jpg"},
        {"name": "Idli Sambar", "ingredients": ["rice", "urad dal", "tamarind", "dal"], "instructions": "Steam idlis and serve with sambar.", "image": "https://www.themealdb.com/images/media/meals/xxpqsy1511452222.jpg"},
    ],
    "Italian": [
        {"name": "Margherita Pizza", "ingredients": ["flour", "cheese", "tomato", "basil"], "instructions": "Bake pizza topped with tomato and mozzarella.", "image": "https://www.themealdb.com/images/media/meals/x0lk931587671540.jpg"},
        {"name": "Pasta Alfredo", "ingredients": ["pasta", "cream", "butter", "cheese"], "instructions": "Cook pasta and toss in creamy Alfredo sauce.", "image": "https://www.themealdb.com/images/media/meals/uquqtu1511178042.jpg"},
    ],
    "Mexican": [
        {"name": "Tacos", "ingredients": ["tortilla", "chicken", "cheese", "lettuce"], "instructions": "Fill tortilla with spiced chicken and veggies.", "image": "https://www.themealdb.com/images/media/meals/wvpsxx1468256321.jpg"},
        {"name": "Quesadilla", "ingredients": ["tortilla", "cheese", "beans"], "instructions": "Grill tortilla with cheese and beans inside.", "image": "https://www.themealdb.com/images/media/meals/1tsqtr1485546232.jpg"},
    ],
    "Japanese": [
        {"name": "Sushi", "ingredients": ["rice", "nori", "fish", "soy sauce"], "instructions": "Roll rice and fish in nori seaweed.", "image": "https://www.themealdb.com/images/media/meals/g046bb1663960946.jpg"},
        {"name": "Ramen", "ingredients": ["noodles", "egg", "broth", "soy sauce"], "instructions": "Boil noodles in flavored broth with egg.", "image": "https://www.themealdb.com/images/media/meals/xqyyqu1511557542.jpg"},
    ],
    "Continental": [
        {"name": "Grilled Chicken", "ingredients": ["chicken", "pepper", "lemon", "garlic"], "instructions": "Grill marinated chicken with herbs.", "image": "https://www.themealdb.com/images/media/meals/wvpsxx1468256321.jpg"},
        {"name": "Caesar Salad", "ingredients": ["lettuce", "croutons", "cheese", "chicken"], "instructions": "Mix lettuce, chicken and Caesar dressing.", "image": "https://www.themealdb.com/images/media/meals/llcbn01574260722.jpg"},
    ]
}

# ---------- CREATE DATAFRAME ----------
all_data = []
for cuisine, recs in recipe_data.items():
    all_data.extend(generate_recipes(cuisine, recs))
df = pd.DataFrame(all_data)

# ---------- TF-IDF ----------
vectorizer = TfidfVectorizer()
vectorizer.fit(df["ingredients"])

# ---------- SESSION ----------
if "favorites" not in st.session_state:
    st.session_state.favorites = []

# ---------- UI ----------
st.markdown("<h1>🍽 Smart Recipe Generator</h1>", unsafe_allow_html=True)
st.write("---")

# Filters
st.sidebar.header("✨ Filters")
cuisine = st.sidebar.selectbox("Select Cuisine", ["All"] + sorted(df["cuisine"].unique()))
search = st.sidebar.text_input("🔍 Search Recipe Name")
ingredients = st.sidebar.text_input("🥕 Enter Ingredients (comma separated)")

# Filter
filtered = df if cuisine == "All" else df[df["cuisine"] == cuisine]

# Search logic
if ingredients.strip():
    vec_user = vectorizer.transform([ingredients])
    vec_rec = vectorizer.transform(filtered["ingredients"])
    sim = cosine_similarity(vec_user, vec_rec).flatten()
    filtered["similarity"] = sim
    results = filtered.sort_values("similarity", ascending=False).head(10)
else:
    results = filtered.copy()

if search.strip():
    results = results[results["recipe"].str.contains(search, case=False)]

if results.empty:
    st.warning("⚠️ No results found! Showing similar recipes...")
    key = ingredients.split(",")[0] if ingredients else search
    results = df[df["ingredients"].str.contains(key, case=False, na=False)].head(5)

# ---------- Display Results ----------
st.subheader("🍴 Recipe Suggestions")

for _, r in results.iterrows():
    st.markdown("<div class='recipe-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(r["image"], use_column_width=True)
    with col2:
        st.markdown(f"### {r['recipe']} ({r['cuisine']})")
        st.markdown(f"⭐ **Rating:** {r['rating']} / 5.0")
        st.markdown(f"🔥 **Difficulty:** {r['difficulty']}")
        st.markdown(f"**Ingredients:** {r['ingredients']}")
        st.markdown(f"**Instructions:** {r['instructions']}")
        if st.button(f"❤️ Add to Favorites", key=r["recipe"]):
            if r["recipe"] not in st.session_state.favorites:
                st.session_state.favorites.append(r["recipe"])
                st.success(f"Added '{r['recipe']}' to favorites!")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

if st.session_state.favorites:
    st.write("---")
    st.subheader("❤️ Your Favorite Recipes:")
    fav_df = df[df["recipe"].isin(st.session_state.favorites)]
    for _, fav in fav_df.iterrows():
        st.markdown(f"- {fav['recipe']} ({fav['cuisine']}) ⭐ {fav['rating']}")

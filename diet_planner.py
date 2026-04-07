import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
from google import genai

load_dotenv()


class DietPlanner:
    def __init__(self):  # ✅ FIXED (was _init_)
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.food_data = {}
        self.api_key = os.getenv("GEMINI_API_KEY")

        # ✅ Gemini client (NEW SDK)
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def load_data(self, data_dir='data'):
        indian_df = pd.read_csv(os.path.join(data_dir, 'Indian_Food.csv'))
        snacks_df = pd.read_csv(os.path.join(data_dir, 'Snacks.csv'))
        fruits_df = pd.read_csv(os.path.join(data_dir, 'fruits.csv'))

        # Merge datasets
        self.food_data['main'] = self._standardize_df(
            pd.concat([indian_df, fruits_df], ignore_index=True)
        )
        self.food_data['snacks'] = self._standardize_df(
            pd.concat([snacks_df, fruits_df], ignore_index=True)
        )

        self._clean_data()

    def _standardize_df(self, df):
        df.columns = df.columns.str.strip()
        df = df.rename(columns={
            'Food': 'name',
            'Class': 'diet_type',
            'Calories': 'calories',
            'Protein': 'protein',
            'Carbs': 'carbs',
            'Fat': 'fat'
        })

        for col in ['calories', 'protein', 'carbs', 'fat']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df['diet_type'] = (
            df['diet_type']
            .astype(str)
            .str.lower()
            .str.replace(" ", "")
            .replace({
                'nonveg': 'non-veg',
                'nonvegetarian': 'non-veg',
                'nv': 'non-veg',
                'veg': 'veg',
                'vegetarian': 'veg',
                'v': 'veg',
                'vegan': 'vegan'
            })
        )

        return df[['name', 'calories', 'protein', 'carbs', 'fat', 'diet_type']]

    def _clean_data(self):
        for key in self.food_data.keys():
            df = self.food_data[key].drop_duplicates(subset='name').copy()
            for col in ['calories', 'protein', 'carbs', 'fat']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(
                    df[col].median()
                )
            self.food_data[key] = df.reset_index(drop=True)

    def _pick_meal_items(self, pool, target_calories):
        pool = pool.sample(frac=1).reset_index(drop=True)
        selected = []
        total_cals = 0

        for _, row in pool.iterrows():
            if total_cals + row['calories'] <= target_calories * 1.1:
                selected.append(row)
                total_cals += row['calories']
            if total_cals >= target_calories * 0.9:
                break

        return selected

    def generate_weekly_plan(
        self,
        daily_calories,
        diet_type='veg',
        meal_counts=None,
        allergens=None,
        options_per_meal=1
    ):
        if meal_counts is None:
            meal_counts = {
                'Breakfast': 0.4,
                'Morning Snack': 0.1,
                'Lunch': 0.4,
                'Evening Snack': 0.1,
                'Dinner': 0.4
            }
        if allergens is None:
            allergens = []

        weekly_plan = []
        used_global = set()

        chapati_keywords = ["chapati", "roti", "fulka", "phulka", "paratha"]

        def is_chapati(item_name):
            return any(k in item_name.lower() for k in chapati_keywords)

        chapati_used = False

        for day in range(7):
            day_plan = {'day': f'Day {day + 1}', 'meals': []}
            used_daily = set()

            for meal_name, ratio in meal_counts.items():
                target = daily_calories * ratio
                pool_type = 'snacks' if 'Snack' in meal_name else 'main'
                pool = self.food_data[pool_type].copy()

                if diet_type != 'all':
                    if diet_type.lower() in ['non vegetarian', 'non-veg']:
                        pool = pool[pool['diet_type'].isin(['veg', 'non-veg'])]
                    else:
                        pool = pool[pool['diet_type'] == diet_type]

                if allergens:
                    pool = pool[
                        ~pool['name'].str.lower().apply(
                            lambda x: any(a.lower() in x for a in allergens)
                        )
                    ]

                pool = pool[~pool['name'].isin(used_global)]
                pool = pool[~pool['name'].isin(used_daily)]

                if chapati_used:
                    pool = pool[~pool['name'].str.lower().apply(is_chapati)]

                if pool.empty:
                    pool = self.food_data[pool_type].copy()

                picks = self._pick_meal_items(pool, target)
                meal_options = []

                for _, row in pd.DataFrame(picks).iterrows():
                    food_name = row['name']
                    used_global.add(food_name)
                    used_daily.add(food_name)

                    if is_chapati(food_name):
                        chapati_used = True

                    meal_options.append({
                        'name': food_name,
                        'calories': float(row['calories']),
                        'protein': float(row['protein']),
                        'carbs': float(row['carbs']),
                        'fat': float(row['fat']),
                        'recipe': None
                    })

                day_plan['meals'].append({
                    'meal_name': meal_name,
                    'target_calories': round(target, 2),
                    'options': meal_options if meal_options else [{
                        'name': 'Fallback Item',
                        'calories': 0,
                        'protein': 0,
                        'carbs': 0,
                        'fat': 0,
                        'recipe': None
                    }]
                })

            weekly_plan.append(day_plan)

        return weekly_plan

    def get_ai_recipe(self, food_name):
        try:
            if not self.client:
                return "API key missing."

            prompt = (
                f"Give a healthy, oil-free recipe for: {food_name}. "
                "Include steps and ingredients. Avoid fried and oily items."
            )

            response = self.client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )

            return getattr(response, "text", "No response").strip()  # ✅ SAFE FIX

        except Exception as e:
            print("Gemini Error:", e)
            return "Recipe generation failed."

    def get_all_ai_recipes_for_day(self, day_meals):
        all_recipes = {}
        for meal in day_meals:
            for option in meal['options']:
                food_name = option['name']
                all_recipes[food_name] = self.get_ai_recipe(food_name)
        return all_recipes
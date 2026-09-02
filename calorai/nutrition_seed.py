"""Hardcoded nutrition table.

Per-unit macros for the foods CalorAI's users actually eat. Values are typical
home-cooked Indian portions drawn from IFCT/USDA-style references and rounded --
they are good to roughly +/-15%, which is well inside the error of a user
saying "two thirds of the box" anyway.

Why a table instead of a nutrition API: the test conversation set is dominated
by Indian home food (paratha, roti, chai, biryani). USDA FoodData Central and
similar APIs are poor on exactly these and would add a network round trip to
the critical path. A local table is a sub-millisecond lookup with better
coverage for this user base. The trade-off is documented in the README.

Fields: (display, unit, kcal, protein_g, carbs_g, fat_g, veg)
"""

SEED_FOODS: dict[str, tuple[str, str, float, float, float, float, int]] = {
    # --- Indian breads ---
    "roti": ("roti / chapati", "piece", 104, 3.1, 20.0, 1.7, 1),
    "phulka": ("phulka", "piece", 85, 2.7, 17.0, 0.7, 1),
    "paratha": ("plain paratha", "piece", 210, 4.4, 27.0, 9.5, 1),
    "aloo paratha": ("aloo paratha", "piece", 290, 6.0, 38.0, 12.5, 1),
    "naan": ("naan", "piece", 260, 8.0, 45.0, 5.0, 1),
    "puri": ("puri", "piece", 140, 2.4, 14.0, 8.0, 1),
    "bhatura": ("bhatura", "piece", 300, 6.0, 40.0, 13.0, 1),
    "dosa": ("plain dosa", "piece", 165, 3.5, 27.0, 4.5, 1),
    "masala dosa": ("masala dosa", "piece", 290, 6.0, 45.0, 9.0, 1),
    "idli": ("idli", "piece", 58, 1.9, 12.0, 0.3, 1),
    "vada": ("medu vada", "piece", 145, 4.0, 15.0, 8.0, 1),
    "bread": ("bread slice", "slice", 75, 2.6, 14.0, 0.9, 1),

    # --- rice and grains ---
    "rice": ("cooked white rice", "cup", 205, 4.3, 45.0, 0.4, 1),
    "brown rice": ("cooked brown rice", "cup", 216, 5.0, 45.0, 1.8, 1),
    "jeera rice": ("jeera rice", "cup", 250, 4.5, 45.0, 6.0, 1),
    "pulao": ("veg pulao", "cup", 260, 5.5, 44.0, 7.0, 1),
    "veg biryani": ("veg biryani", "cup", 290, 6.5, 45.0, 9.0, 1),
    "chicken biryani": ("chicken biryani", "cup", 350, 17.0, 42.0, 13.0, 0),
    "biryani": ("biryani (mixed)", "cup", 320, 11.0, 44.0, 11.0, 1),
    "poha": ("poha", "cup", 250, 5.0, 45.0, 6.0, 1),
    "upma": ("upma", "cup", 230, 5.5, 38.0, 7.0, 1),
    "khichdi": ("khichdi", "cup", 210, 8.0, 34.0, 4.5, 1),
    "oats": ("cooked oats", "cup", 160, 6.0, 27.0, 3.2, 1),
    "poori sabzi": ("poori with sabzi", "plate", 480, 9.0, 58.0, 24.0, 1),

    # --- dals, legumes, curries ---
    "dal": ("dal (tadka)", "cup", 180, 9.0, 24.0, 5.0, 1),
    "rajma": ("rajma curry", "cup", 230, 11.0, 33.0, 6.0, 1),
    "chole": ("chole", "cup", 260, 11.0, 35.0, 8.5, 1),
    "sambar": ("sambar", "cup", 140, 6.5, 20.0, 4.0, 1),
    "chana": ("boiled chana", "cup", 210, 11.0, 35.0, 3.5, 1),
    "paneer curry": ("paneer curry", "cup", 320, 14.0, 12.0, 24.0, 1),
    "paneer": ("paneer", "100g", 265, 18.0, 3.4, 20.0, 1),
    "palak paneer": ("palak paneer", "cup", 290, 13.0, 11.0, 22.0, 1),
    "aloo sabzi": ("aloo sabzi", "cup", 180, 3.5, 26.0, 7.0, 1),
    "mixed veg": ("mixed veg sabzi", "cup", 150, 4.0, 18.0, 7.0, 1),
    "bhindi": ("bhindi masala", "cup", 160, 3.0, 15.0, 10.0, 1),

    # --- non-veg ---
    "chicken curry": ("chicken curry", "cup", 290, 25.0, 8.0, 18.0, 0),
    "butter chicken": ("butter chicken", "cup", 380, 26.0, 12.0, 26.0, 0),
    "chicken breast": ("grilled chicken breast", "100g", 165, 31.0, 0.0, 3.6, 0),
    "egg": ("whole egg", "piece", 72, 6.3, 0.4, 4.8, 0),
    "omelette": ("2-egg omelette", "serving", 190, 13.0, 1.5, 14.0, 0),
    "boiled egg": ("boiled egg", "piece", 72, 6.3, 0.4, 4.8, 0),
    "fish curry": ("fish curry", "cup", 240, 22.0, 7.0, 13.0, 0),
    "mutton curry": ("mutton curry", "cup", 350, 24.0, 6.0, 25.0, 0),

    # --- drinks ---
    "chai": ("chai (with milk & sugar)", "cup", 105, 2.5, 13.0, 4.5, 1),
    "black coffee": ("black coffee", "cup", 5, 0.3, 0.0, 0.0, 1),
    "coffee": ("coffee with milk", "cup", 90, 3.0, 10.0, 4.0, 1),
    "milk": ("whole milk", "cup", 150, 8.0, 12.0, 8.0, 1),
    "lassi": ("sweet lassi", "glass", 220, 7.0, 32.0, 6.0, 1),
    "buttermilk": ("chaas", "glass", 60, 3.0, 6.0, 2.5, 1),
    "orange juice": ("orange juice", "glass", 110, 1.7, 26.0, 0.5, 1),
    "protein shake": ("whey protein shake", "scoop", 120, 24.0, 3.0, 1.5, 1),

    # --- snacks & sweets ---
    "samosa": ("samosa", "piece", 260, 4.0, 30.0, 14.0, 1),
    "pakora": ("pakora", "piece", 75, 2.0, 7.0, 4.5, 1),
    "biscuit": ("tea biscuit", "piece", 45, 0.7, 7.0, 1.7, 1),
    "banana": ("banana", "piece", 105, 1.3, 27.0, 0.4, 1),
    "apple": ("apple", "piece", 95, 0.5, 25.0, 0.3, 1),
    "almonds": ("almonds", "10 pieces", 70, 2.6, 2.5, 6.0, 1),
    "curd": ("curd / dahi", "cup", 150, 8.5, 11.0, 8.0, 1),
    "gulab jamun": ("gulab jamun", "piece", 150, 2.0, 22.0, 6.0, 1),
    "ice cream": ("ice cream", "scoop", 140, 2.5, 17.0, 7.0, 1),
    "chips": ("potato chips", "small packet", 150, 2.0, 15.0, 10.0, 1),
    "maggi": ("maggi noodles", "packet", 310, 7.0, 43.0, 12.0, 1),
    "pizza": ("pizza slice", "slice", 285, 12.0, 36.0, 10.0, 1),
    "burger": ("veg burger", "piece", 350, 12.0, 42.0, 15.0, 1),
    "sandwich": ("veg sandwich", "piece", 280, 9.0, 38.0, 10.0, 1),
    "salad": ("green salad", "bowl", 60, 2.0, 10.0, 1.5, 1),
    "dal khichdi": ("dal khichdi", "cup", 220, 8.5, 35.0, 5.0, 1),
}

import os
import re
import numpy as np
import tensorflow as tf

# --- GLOBAL CONFIG ---
IMG_SIZE = (256, 256)
CATEGORIES = ["group1", "group2", "group3", "group4"]

# --- NORWOOD REFINEMENT & GRAFT ESTIMATION ---
def refined_norwood_prediction(group: int, age: int, smokes: bool, race: str) -> str:
    """
    Refines the Norwood stage prediction within a given group based on normalized 
    age, smoking status, and race.
    """
    normalized_age = (age - 39.67) / 17.73

    race_scores = {
        'Asian': 0.183,
        'Caucasian': -0.04,
        'Black': -0.145
    }

    base_score = (
        normalized_age * 0.88
        + (0.64 if smokes else 0)
        + race_scores.get(race, 0)
    )

    thresholds = {
        1: -0.92,
        2: -0.5,
        3: 1.0,
        4: 1.75
    }

    if group == 1:
        return "Norwood Stage 1"
    elif group == 2:
        return "Norwood Stage 3" if base_score > thresholds[2] else "Norwood Stage 2"
    elif group == 3:
        return "Norwood Stage 5" if base_score > thresholds[3] else "Norwood Stage 4"
    elif group == 4:
        return "Norwood Stage 7" if base_score > thresholds[4] else "Norwood Stage 6"
    else:
        return "Unknown Group"

def estimate_grafts(norwood_stage: str, age: int) -> int:
    """
    Estimate the number of grafts needed based on Norwood stage and age (ages
    between 25 and 70 are interpolated).
    """
    graft_ranges = {
        "II":  (800, 1200),
        "III": (1000, 1500),
        "IV":  (1600, 2200),
        "V":   (2000, 2700),
        "VI":  (2500, 3500),
        "VII": (3500, 5000),
    }

    if norwood_stage not in graft_ranges:
        raise ValueError(f"Norwood stage '{norwood_stage}' is not recognized in graft estimation.")

    if age < 18:
        raise ValueError("Age is too low for typical hair transplant recommendations.")

    min_grafts, max_grafts = graft_ranges[norwood_stage]

    young_age = 25
    old_age = 70
    interpolation_factor = min(max((age - young_age) / (old_age - young_age), 0), 1)
    estimated_grafts = min_grafts + interpolation_factor * (max_grafts - min_grafts)

    return int(round(estimated_grafts))

def combine_prediction_and_estimation(group: int, age: int, smokes: bool, race: str):
    """
    Combines the refined Norwood prediction with the graft estimation (if needed).
    """
    refined_stage_str = refined_norwood_prediction(group, age, smokes, race)

    # Norwood Stage 1 => no grafts needed
    if refined_stage_str == "Norwood Stage 1":
        return refined_stage_str, "No visible hair loss, zero grafts needed."

    # Extract the numeric portion from "Norwood Stage X"
    match = re.search(r"(\d+)$", refined_stage_str)
    if not match:
        return refined_stage_str, None

    numeric_stage = int(match.group(1))
    norwood_map = {
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII"
    }
    roman_stage = norwood_map.get(numeric_stage)
    if roman_stage is None:
        return refined_stage_str, None

    graft_est = estimate_grafts(roman_stage, age)
    return refined_stage_str, graft_est

def print_cost_table(grafts):
    countries = {
        'United States': {'low': 2, 'high': 10},
        'Canada': {'low': 1.85, 'high': 5.18},
        'United Kingdom': {'low': 4.88, 'high': 4.88},  # Single value as low and high are the same
        'Australia': {'low': 1.28, 'high': 2.94},
        'Singapore': {'low': 2.88, 'high': 4.32},
        'Turkey': {'low': 0.58, 'high': 2.63},
        'India': {'low': 0.78, 'high': 0.78}  # Single value as low and high are the same
    }
    print("\nCost Table for Estimated Grafts:")
    print("| Country         | Cost per Graft (USD) | Low Total Cost (USD)  | High Total Cost (USD)  | Avg Total Cost (USD) |")
    print("|-----------------|----------------------|-----------------------|------------------------|----------------------|")
    for country, prices in countries.items():
        low_cost = prices['low'] * grafts
        high_cost = prices['high'] * grafts
        avg_cost_per_graft = (prices['low'] + prices['high']) / 2
        avg_total_cost = avg_cost_per_graft * grafts
        cost_per_graft_range = f"${prices['low']:.2f} to ${prices['high']:.2f}" if prices['low'] != prices['high'] else f"${prices['low']:.2f}"
        
        print(f"| {country:15} | {cost_per_graft_range:20} | ${low_cost:20,.0f} | ${high_cost:21,.0f} | ${avg_total_cost:19,.0f} |")


# --- MODEL LOADING & PREDICTION ---
def load_model(model_filename: str) -> tf.keras.Model:
    """
    Loads a TensorFlow Keras model from disk.
    """
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    return tf.keras.models.load_model(model_filename)

def predict_image(model: tf.keras.Model, image_path: str) -> tuple[str, float]:
    """
    Predict the class of a given image using the loaded model.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=IMG_SIZE, 
            color_mode="grayscale"
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        return CATEGORIES[predicted_idx], confidence

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None

def test_all_images_with_estimation(test_images_directory: str, model: tf.keras.Model) -> None:
    """
    For each image in test_images_directory, prompt the user to provide
    age, smoking status, and race data. Then make a prediction and estimate grafts.
    """
    if not os.path.exists(test_images_directory):
        print(f"Test images folder not found: {test_images_directory}")
        return

    for filename in os.listdir(test_images_directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(test_images_directory, filename)

            print(f"\n--- Analyzing image '{filename}' ---")

            # Age input
            while True:
                try:
                    age = int(input("Enter the age of the person for this image: ").strip())
                    break
                except ValueError:
                    print("Invalid age. Please enter an integer.")

            # Smoking input
            smokes = input("Does the person smoke? (y/n): ").strip().lower() == 'y'

            # Race input
            race_input = input("Enter the race (a: Asian, b: Black, c: Caucasian): ").strip().lower()
            race_map = {'a': 'Asian', 'b': 'Black', 'c': 'Caucasian'}
            race = race_map.get(race_input, 'Unknown')

            predicted_class, confidence = predict_image(model, image_path)

            if predicted_class:
                group_num_str = re.sub(r"\D", "", predicted_class)
                if not group_num_str:
                    print(f"Unable to parse group from predicted class: {predicted_class}")
                    continue

                group = int(group_num_str)
                refined_stage, grafts = combine_prediction_and_estimation(group, age, smokes, race)

                print(f"Predicted Class:        {predicted_class}")
                print(f"Confidence:             {confidence:.2f}%")
                print(f"Refined Norwood Stage:  {refined_stage}")

                if isinstance(grafts, int):
                    print(f"Estimated Grafts:       {grafts}")
                    print_cost_table(grafts)
                else:
                    print(f"Message:                {grafts}")
            else:
                print(f"Prediction failed for image: {filename}")

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    try:
        data_directory = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(data_directory, "fe4og.keras")
        test_images_directory = os.path.join(data_directory, "test_images")

        model = load_model(model_path)

        test_all_images_with_estimation(test_images_directory, model)
    except Exception as e:
        print(f"An error occurred: {e}")
    
# Adapted from https://github.com/LwbXc/STBench/blob/main/code/result_parser.py
# But only do model answer parsing here (evaluation logic is in STBench.py)

from geopy.distance import geodesic
import re


def find_last_digit(response):
    for char in reversed(response):
        if char.isdigit():
            return char
    return None


def trajectory_classification(response):
    pattern = r"car|bike|bicycle|pedestrian"
    mapping = {"car": 1, "bike": 2, "bicycle": 2, "pedestrian": 3}
    match = re.search(pattern, response, flags=re.I)
    if match:
        predicted = match.group()
        predicted = mapping[predicted]
        return predicted
    return None


def yes_or_no(response):
    pattern = r"\b(Yes|No)\b"
    match = re.search(pattern, response, flags=re.I)
    if match:
        predicted = match.group()
        predicted = predicted.title()
        return predicted
    return None


def anomaly_detection(response):
    pattern = r"Normal|Anomalous|Anomaly|Abnormal"
    match = re.search(pattern, response, flags=re.I)
    if match:
        predicted = match.group()
        predicted = predicted.title()
        if predicted == "Abnormal" or predicted == "Anomaly":
            predicted = "Anomalous"
        return predicted.lower()
    return None


def extract_coordinates(text: str):
    lon_min, lon_max = 105, 110
    lat_min, lat_max = 30, 36

    matches = re.findall(
        r"\(([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)(?:,\s*[-+]?\d+)?\)", text
    )
    for lon, lat in matches[::-1]:
        lon, lat = float(lon), float(lat)
        if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
            return (lon, lat)

    nums = re.findall(r"[-+]?\d+\.\d+", text)
    nums = list(map(float, nums))

    coord = (None, None)
    for i in range(len(nums) - 2, -1, -1):
        if lon_min <= nums[i] <= lon_max and coord[0] == None:
            coord[0] = nums[i]
        if lat_min <= nums[i] <= lat_max and coord[1] == None:
            coord[1] = nums[i]
    if coord[0] and coord[1]:
        return coord

    return None


def calculate_distance(coord1, coord2):
    distance = geodesic([coord2[1], coord2[0]], [coord1[1], coord1[0]]).meters
    return distance


def trajectory_prediction(response):
    try:
        lon, lat = extract_coordinates(response)
    except Exception:
        return None
    return (lon, lat)


def flow_prediction(response):
    try:
        seq_pattern = r"(\[?\s*-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*){5}\s*\]?)"
        seq_match = re.search(seq_pattern, response)
        if not seq_match:
            return None

        seq_str = seq_match.group(1).strip("[]")
        predicted = [float(x) for x in seq_str.split(",")]
        return predicted

    except Exception:
        return None


result_parser = {
    "administrative_region_determination": find_last_digit,
    "poi_category_recognition": find_last_digit,
    "poi_identification": yes_or_no,
    "urban_region_function_recognition": find_last_digit,
    "point_trajectory": find_last_digit,
    "point_region": find_last_digit,
    "trajectory_region": find_last_digit,
    "trajectory_identification": yes_or_no,
    "trajectory_trajectory": find_last_digit,
    "direction_determination": find_last_digit,
    "trajectory_anomaly_detection": anomaly_detection,
    "trajectory_classification": trajectory_classification,
    "trajectory_prediction": trajectory_prediction,
    "flow_prediction": flow_prediction,
    "navigation": find_last_digit,
    "road_level_judgment": find_last_digit,
    "rush_hour_detection": yes_or_no,
    "taxi_occupancy_detection": yes_or_no,
}

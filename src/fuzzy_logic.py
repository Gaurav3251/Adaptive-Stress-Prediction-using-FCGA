import numpy as np
from typing import Dict, Tuple, List

class FuzzyStressPredictor:
    """Fuzzy Logic System for Stress Prediction"""
    
    def __init__(self, rule_weights=None):
        """Initialize with optional GA-optimized rule weights"""
        self.rule_weights = rule_weights if rule_weights is not None else np.ones(15)
    
    def membership_sleep(self, hours: float) -> Dict[str, float]:
        """Fuzzy membership for sleep duration"""
        return {
            'low': max(0, min(1, (5 - hours) / 2)),
            'medium': max(0, min((hours - 4) / 2, (8 - hours) / 2)),
            'high': max(0, min(1, (hours - 6) / 2))
        }
    
    def membership_sleep_quality(self, quality: int) -> Dict[str, float]:
        """Fuzzy membership for sleep quality"""
        return {
            'poor': max(0, min(1, (3 - quality) / 2)),
            'fair': 1.0 if quality == 3 else max(0, 1 - abs(quality - 3) / 2),
            'good': max(0, min(1, (quality - 2) / 2))
        }
    
    def membership_work_hours(self, hours: float) -> Dict[str, float]:
        """Fuzzy membership for work hours"""
        return {
            'normal': max(0, min(1, (9 - hours) / 2)),
            'high': max(0, min((hours - 7) / 3, (12 - hours) / 3)),
            'excessive': max(0, min(1, (hours - 10) / 3))
        }
    
    def membership_activity(self, hours: float) -> Dict[str, float]:
        """Fuzzy membership for physical activity"""
        return {
            'low': max(0, min(1, (2 - hours) / 2)),
            'medium': max(0, min((hours - 0.5) / 2, (4 - hours) / 2)),
            'high': max(0, min(1, (hours - 2) / 3))
        }
    
    def membership_screen_time(self, hours: float) -> Dict[str, float]:
        """Fuzzy membership for screen time"""
        return {
            'low': max(0, min(1, (3 - hours) / 2)),
            'medium': max(0, min((hours - 2) / 2, (5 - hours) / 2)),
            'high': max(0, min(1, (hours - 4) / 2))
        }
    
    def membership_health_risk(self, bp: int, chol: int, sugar: int) -> Dict[str, float]:
        """Fuzzy membership for health risk based on BP, cholesterol, sugar"""
        # Normalize health metrics
        bp_risk = (bp - 120) / 30  # Normal around 120
        chol_risk = (chol - 200) / 50  # Normal around 200
        sugar_risk = (sugar - 100) / 30  # Normal around 100
        
        avg_risk = (bp_risk + chol_risk + sugar_risk) / 3
        
        return {
            'low': max(0, min(1, -avg_risk + 0.5)),
            'medium': 1.0 - abs(avg_risk),
            'high': max(0, min(1, avg_risk + 0.5))
        }
    
    def apply_fuzzy_rules(self, memberships: Dict) -> Dict[str, float]:
        """Apply fuzzy rules with GA-optimized weights"""
        stress_scores = {'Low': 0, 'Medium': 0, 'High': 0}
        
        # Rule 1: Low sleep AND high work -> High stress
        stress_scores['High'] += self.rule_weights[0] * min(
            memberships['sleep']['low'],
            memberships['work']['high']
        )
        
        # Rule 2: Good sleep AND low work -> Low stress
        stress_scores['Low'] += self.rule_weights[1] * min(
            memberships['sleep']['high'],
            memberships['work']['normal']
        )
        
        # Rule 3: High activity AND meditation -> Low stress
        stress_scores['Low'] += self.rule_weights[2] * min(
            memberships['activity']['high'],
            memberships['meditation']
        )
        
        # Rule 4: High screen time AND low activity -> High stress
        stress_scores['High'] += self.rule_weights[3] * min(
            memberships['screen']['high'],
            memberships['activity']['low']
        )
        
        # Rule 5: Poor sleep quality -> High stress
        stress_scores['High'] += self.rule_weights[4] * memberships['sleep_quality']['poor']
        
        # Rule 6: Excessive work hours -> High stress
        stress_scores['High'] += self.rule_weights[5] * memberships['work']['excessive']
        
        # Rule 7: High health risk -> High stress
        stress_scores['High'] += self.rule_weights[6] * memberships['health']['high']
        
        # Rule 8: Low social interaction -> Medium stress
        stress_scores['Medium'] += self.rule_weights[7] * memberships['social_low']
        
        # Rule 9: Medium sleep AND medium work -> Medium stress
        stress_scores['Medium'] += self.rule_weights[8] * min(
            memberships['sleep']['medium'],
            memberships['work']['high']
        )
        
        # Rule 10: High caffeine -> Medium stress
        stress_scores['Medium'] += self.rule_weights[9] * memberships['caffeine_high']
        
        # Rule 11: Smoking AND alcohol -> High stress
        stress_scores['High'] += self.rule_weights[10] * min(
            memberships['smoking'],
            memberships['alcohol_high']
        )
        
        # Rule 12: Good activity AND good sleep -> Low stress
        stress_scores['Low'] += self.rule_weights[11] * min(
            memberships['activity']['medium'],
            memberships['sleep']['high']
        )
        
        # Rule 13: Long travel time -> Medium stress
        stress_scores['Medium'] += self.rule_weights[12] * memberships['travel_high']
        
        # Rule 14: Low health risk AND meditation -> Low stress
        stress_scores['Low'] += self.rule_weights[13] * min(
            memberships['health']['low'],
            memberships['meditation']
        )
        
        # Rule 15: Multiple medium factors -> Medium stress
        stress_scores['Medium'] += self.rule_weights[14] * min(
            memberships['screen']['medium'],
            memberships['work']['high']
        )
        
        return stress_scores
    
    def defuzzify(self, stress_scores: Dict[str, float]) -> Tuple[str, float]:
        """Convert fuzzy output to crisp prediction"""
        total = sum(stress_scores.values())
        
        if total == 0:
            return "Medium", 50.0
        
        # Normalize scores
        normalized = {k: v/total for k, v in stress_scores.items()}
        
        # Get prediction and confidence
        prediction = max(normalized, key=normalized.get)
        confidence = normalized[prediction] * 100
        
        return prediction, confidence
    
    def predict(self, input_data: Dict) -> Tuple[str, float]:
        """Main prediction function"""
        # Extract features with proper type conversion
        sleep_dur = float(input_data.get('Sleep_Duration', 7))
        sleep_qual = int(input_data.get('Sleep_Quality', 4))
        work_hours = float(input_data.get('Work_Hours', 8))
        activity = float(input_data.get('Physical_Activity', 2))
        screen = float(input_data.get('Screen_Time', 4))
        bp = int(input_data.get('Blood_Pressure', 120))
        chol = int(input_data.get('Cholesterol_Level', 200))
        sugar = int(input_data.get('Blood_Sugar_Level', 90))
        
        # Convert meditation to numeric if it's a string
        meditation_val = input_data.get('Meditation_Practice', 0)
        if isinstance(meditation_val, str):
            meditation = 1 if meditation_val.lower() in ['yes', '1', 'true'] else 0
        else:
            meditation = int(meditation_val)
        
        social = int(input_data.get('Social_Interactions', 3))
        caffeine = int(input_data.get('Caffeine_Intake', 1))
        alcohol = int(input_data.get('Alcohol_Intake', 0))
        
        # Convert smoking to numeric if it's a string
        smoking_val = input_data.get('Smoking_Habit', 0)
        if isinstance(smoking_val, str):
            smoking = 1 if smoking_val.lower() in ['yes', '1', 'true'] else 0
        else:
            smoking = int(smoking_val)
        
        travel = float(input_data.get('Travel_Time', 1))
        
        # Calculate fuzzy memberships
        memberships = {
            'sleep': self.membership_sleep(sleep_dur),
            'sleep_quality': self.membership_sleep_quality(sleep_qual),
            'work': self.membership_work_hours(work_hours),
            'activity': self.membership_activity(activity),
            'screen': self.membership_screen_time(screen),
            'health': self.membership_health_risk(bp, chol, sugar),
            'meditation': meditation,
            'social_low': 1.0 if social < 3 else 0.0,
            'caffeine_high': 1.0 if caffeine > 2 else 0.0,
            'alcohol_high': 1.0 if alcohol > 1 else 0.0,
            'smoking': smoking,
            'travel_high': 1.0 if travel > 2 else 0.0
        }
        
        # Apply fuzzy rules
        stress_scores = self.apply_fuzzy_rules(memberships)
        
        # Defuzzify
        return self.defuzzify(stress_scores)
    
    def get_recommendations(self, input_data: Dict, stress_level: str) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if input_data['Sleep_Duration'] < 7:
            recommendations.append("Increase sleep duration to at least 7-8 hours")
        
        if input_data['Work_Hours'] > 9:
            recommendations.append("Try to reduce work hours and take regular breaks")
        
        if input_data['Physical_Activity'] < 2:
            recommendations.append("Increase physical activity to at least 2-3 hours per week")
        
        if input_data['Screen_Time'] > 5:
            recommendations.append("Reduce screen time, especially before bedtime")
        
        if input_data['Meditation_Practice'] == 0:
            recommendations.append("Consider starting meditation or mindfulness practice")
        
        if input_data['Social_Interactions'] < 3:
            recommendations.append("Increase social interactions with friends and family")
        
        if input_data['Caffeine_Intake'] > 2:
            recommendations.append("Reduce caffeine intake, especially in the afternoon")
        
        if stress_level == "High":
            recommendations.append("Consider consulting a healthcare professional")
        
        return recommendations if recommendations else ["Maintain your current healthy lifestyle!"]
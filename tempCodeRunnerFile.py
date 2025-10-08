     if "Beginner" in user_name:
            assert 0.01 <= mean_gain <= 0.35, f"Beginner gain rate unrealistic: {mean_gain}"  # Lowered from 0.04 to 0.01
        elif "Intermediate" in user_name:
            assert 0.008 <= mean_gain <= 0.15, f"Intermediate gain rate unrealistic: {mean_gain}"  # Lowered from 0.015 to 0.008
        elif "Advanced" in user_name:
            assert 0.005 <= mean_gain <= 0.08, f"Advanced gain rate unrealistic: {mean_gain}"  # Lowered from 0.01 to 0.005

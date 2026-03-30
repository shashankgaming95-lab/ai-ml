import pandas as pd
import numpy as np

# Function to generate student records

def generate_student_data(num_records=200):
    np.random.seed(0)  # For reproducibility
    data = []
    
    for _ in range(num_records):
        hours_studied = np.random.randint(0, 13)
        sleep_hours = np.random.randint(3, 10)
        attendance_percentage = np.random.randint(40, 101)
        previous_grade = np.random.randint(40, 101)
        pass_ = int((hours_studied > 5) + (sleep_hours > 5) + 
                     (attendance_percentage > 70) + (previous_grade > 50) >= 3)
        pass_ += np.random.choice([-1, 0, 1])  # Adding some noise
        data.append([hours_studied, sleep_hours, attendance_percentage, previous_grade, pass_])
    
    columns = ['Hours_Studied', 'Sleep_Hours', 'Attendance_Percentage', 'Previous_Grade', 'Pass']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('student_data.csv', index=False)

# Generate data
if __name__ == '__main__':
    generate_student_data()
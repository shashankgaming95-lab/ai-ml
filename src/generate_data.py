import numpy as np
import pandas as pd

np.random.seed(42)
n_students = 200

hours_studied = np.random.uniform(0, 12, n_students)
sleep_hours = np.random.uniform(3, 9, n_students)
attendance = np.random.uniform(40, 100, n_students)
previous_grade = np.random.uniform(40, 100, n_students)

cond1 = (hours_studied > 5).astype(int)
cond2 = (sleep_hours > 5).astype(int)
cond3 = (attendance > 70).astype(int)
cond4 = (previous_grade > 50).astype(int)

pass_label = cond1 + cond2 + cond3 + cond4
pass_label = (pass_label >= 3).astype(int) + np.random.binomial(1, 0.1, n_students)
pass_label = np.clip(pass_label, 0, 1)

df = pd.DataFrame({
    'Hours_Studied': hours_studied,
    'Sleep_Hours': sleep_hours,
    'Attendance_Percentage': attendance,
    'Previous_Grade': previous_grade,
    'Pass': pass_label
})

df.to_csv('data/student_data.csv', index=False)
print("Dataset saved to data/student_data.csv")
print(df.head())
print("\nClass distribution:\n", df['Pass'].value_counts())

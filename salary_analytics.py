import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("charts/employee_salary_workforce_analytics.csv")

print("=" * 50)
print("EMPLOYEE SALARY DATASET")
print("=" * 50)
print(df.head())
print(df.tail())
print("=" * 50)
print("\nDataset Shape:")
print(df.shape)
print("\ndataset information:")
print(df.info())


print("\nColumn Names:")
print(df.columns)


print("\nBasic Statistical Information:")
print(df.describe())


print("\nMissing Values:")
print(df.isnull().sum())


print("\nDuplicate Records:")
print(df.duplicated().sum())

df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
print("\nMissing Values in Salary Column After Imputation:")

df['Department'] = df['Department'].fillna("Unknown")
print("\nMissing Values in Department Column After Imputation:")

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")

df['Department'] = df['Department'].str.strip()
print("\nUnique Departments in the Dataset:")

df["Gender"] = df["Gender"].str.title()
print("\nUnique Genders in the Dataset:")

df.to_csv("charts/cleaned_employee_salary_workforce_analytics.csv", index=False)

#number of employees
print("total number of employees :",len(df))

# Average Salary
print("Average Salary:", df["Salary"].mean())

# Highest Salary
print("Highest Salary:", df["Salary"].max())

# Lowest Salary
print("Lowest Salary:", df["Salary"].min())

# Calculate Median Salary
print("Median Salary:", df["Salary"].median())

# Average employee age
print("Average employee age:", df["Age"].mean())

# Count employees department-wise
print(df["Department"].value_counts())

# Count employees city-wise
print(df["City"].value_counts())

# Count male & female employees
print(df["Gender"].value_counts())

# Average salary by department
department_salary = df.groupby("Department")["Salary"].mean()

# Display departments in descending order of average salary
print(department_salary.sort_values(ascending=False))

# Department with the highest average salary
print(department_salary.idxmax())

# Highest average salary value
print(department_salary.max())

# Top 10 highest paid employees
top_10 = df.sort_values(by="Salary", ascending=False)
print(top_10.head(10))

# Remote workers
remote = df[df["Work_Mode"] == "Remote"]
print(remote)

# Employees with Performance Score > 8
performance = df[df["Performance_Score"] > 8]
print(performance)

# Average Bonus by Department
bonus = df.groupby("Department")["Bonus"].mean()
print(bonus)

# Average Job Satisfaction by Department
satisfaction = df.groupby("Department")["Job_Satisfaction"].mean()
print(satisfaction)

# Employees who completed more than 5 projects
project = df[df["Projects_Completed"] > 5]
print(project)

# Employees with more than 10 training hours
training = df[df["Training_Hours"] > 10]
print(training)

# Data Visualization
# 1. Gender Distribution (Pie Chart)

gender_count = df["Gender"].value_counts()

plt.figure(figsize=(7,7))
plt.pie(
    gender_count.values,
    labels=gender_count.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Gender Distribution")
plt.savefig("output/gender_distribution.png")
plt.show()


# 2. Department-wise Employee Count (Bar Chart)
department_count = df["Department"].value_counts()

plt.figure(figsize=(8,5))
department_count.plot(kind="bar", color="skyblue")
plt.title("Department-wise Employee Count")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.tight_layout()
plt.savefig("output/department_count.png")
plt.show()


# 3. Salary Distribution (Histogram)
plt.figure(figsize=(8,5))
plt.hist(df["Salary"], bins=10, color="green", edgecolor="black")
plt.title("Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/salary_distribution.png")
plt.show()


# 4. Salary vs Experience (Scatter Plot)
plt.figure(figsize=(8,5))
plt.scatter(df["Experience"], df["Salary"], color="red")
plt.title("Salary vs Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.tight_layout()
plt.savefig("output/salary_vs_experience.png")
plt.show()


# 5. Average Salary by Department (Bar Plot)
avg_salary = df.groupby("Department")["Salary"].mean()
plt.figure(figsize=(8,5))
avg_salary.plot(kind="bar", color="orange")
plt.title("Average Salary by Department")
plt.xlabel("Department")
plt.ylabel("Average Salary")
plt.tight_layout()
plt.savefig("output/average_salary_department.png")
plt.show()


# 6. Job Satisfaction Distribution (Box Plot)
plt.figure(figsize=(6,5))
plt.boxplot(df["Job_Satisfaction"])
plt.title("Job Satisfaction Distribution")
plt.ylabel("Job Satisfaction")
plt.tight_layout()
plt.savefig("output/job_satisfaction_boxplot.png")
plt.show()


# 7. Performance Score Distribution (Histogram)
plt.figure(figsize=(8,5))
plt.hist(df["Performance_Score"], bins=10,
         color="purple", edgecolor="black")
plt.title("Performance Score Distribution")
plt.xlabel("Performance Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("output/performance_score_distribution.png")
plt.show()


# 8. Work Mode Distribution (Count Plot)
plt.figure(figsize=(7,5))
sns.countplot(data=df, x="Work_Mode")
plt.title("Work Mode Distribution")
plt.xlabel("Work Mode")
plt.ylabel("Employees")
plt.tight_layout()
plt.savefig("output/work_mode_distribution.png")
plt.show()

# 9. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")
plt.show()


# 10. Top 10 Highest Salaries (Horizontal Bar Chart)
top_salary = df.nlargest(10, "Salary")
plt.figure(figsize=(10,6))
plt.barh(
    top_salary["Employee_Name"],
    top_salary["Salary"],
    color="brown"
)
plt.title("Top 10 Highest Salaries")
plt.xlabel("Salary")
plt.ylabel("Employee Name")
plt.tight_layout()
plt.savefig("output/top10_highest_salaries.png")
plt.show()


print("\nCharts Generated Successfully")
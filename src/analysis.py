import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from tabulate import tabulate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

# Set style for better visualizations
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

# Read datasets
df1 = pd.read_csv('data/health_dataset1.csv')
df2 = pd.read_csv('data/health_dataset2.csv')

# Print column names to verify
print("\n=== Dataset 1 Columns ===")
print(df1.columns.tolist())
print("\n=== Dataset 2 Columns ===")
print(df2.columns.tolist())

# Function to save plots
def save_plot(name):
    plt.tight_layout()
    plt.savefig(f'plots/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_tabulated(data, headers=None, title=None):
    """Print data in a nicely formatted table"""
    if title:
        print(f"\n=== {title} ===")
    print(tabulate(data, headers=headers, tablefmt='grid', floatfmt='.2f'))
    print()

# 1. Basic Statistical Analysis
print("\n=== Dataset 1 Summary Statistics ===")
summary_stats = df1.describe()
print_tabulated(summary_stats, headers='keys', title='Numerical Variables Summary')

print("\n=== Dataset 1 Missing Values ===")
missing_data = df1.isnull().sum()
print_tabulated(missing_data.to_frame(), headers=['Column', 'Missing Count'])

# 2. Demographic Analysis
# Age distribution
age_bins = pd.cut(df1['Age'], bins=10)
age_distribution = age_bins.value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.histplot(data=df1, x='Age', bins=30, kde=True)
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Count')
save_plot('age_distribution')
print_tabulated(age_distribution.reset_index(), 
                headers=['Age Range', 'Count'],
                title='Age Distribution')

# Add age-related statistical tests
print("\n=== Age Analysis ===")
age_bp_ttest = stats.ttest_ind(
    df1[df1['Blood_Pressure_Abnormality'] == 0]['Age'],
    df1[df1['Blood_Pressure_Abnormality'] == 1]['Age']
)
print(f"T-test for Age difference between BP groups:")
print(f"t-statistic: {age_bp_ttest.statistic:.4f}")
print(f"p-value: {age_bp_ttest.pvalue:.4f}")

# Gender analysis
gender_bp = pd.crosstab(df1['Sex'], df1['Blood_Pressure_Abnormality'])
plt.figure(figsize=(10, 6))
gender_bp.plot(kind='bar', stacked=True)
plt.title('Blood Pressure Abnormality by Gender')
plt.xlabel('Gender (0=Male, 1=Female)')
plt.ylabel('Count')
plt.legend(['Normal', 'Abnormal'])
save_plot('bp_by_gender')
print_tabulated(gender_bp, 
                headers=['Normal BP', 'Abnormal BP'],
                title='Blood Pressure by Gender')

# Chi-square test for gender and BP
gender_bp_chi2 = stats.chi2_contingency(gender_bp)
print("\n=== Gender and BP Chi-square Test ===")
print(f"Chi-square statistic: {gender_bp_chi2[0]:.4f}")
print(f"p-value: {gender_bp_chi2[1]:.4f}")

# 3. BMI Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Blood_Pressure_Abnormality', y='BMI', data=df1)
plt.title('BMI Distribution by Blood Pressure Status')
plt.xlabel('Blood Pressure Abnormality (0=Normal, 1=Abnormal)')
plt.ylabel('BMI')
save_plot('bmi_bp_boxplot')

bmi_stats = df1.groupby('Blood_Pressure_Abnormality')['BMI'].describe()
print_tabulated(bmi_stats, headers='keys', title='BMI Statistics by BP Status')

# BMI categories analysis
df1['BMI_Category'] = pd.cut(df1['BMI'], 
                            bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
bmi_category_stats = pd.crosstab(df1['BMI_Category'], df1['Blood_Pressure_Abnormality'])
print_tabulated(bmi_category_stats, 
                headers=['Normal BP', 'Abnormal BP'],
                title='BMI Categories vs Blood Pressure')

# 4. Correlation Analysis
correlation_matrix = df1.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Variables')
save_plot('correlation_matrix')
print_tabulated(correlation_matrix.round(2), 
                headers='keys',
                title='Correlation Matrix')

# 5. Lifestyle Factors Analysis
# Smoking and Blood Pressure
smoking_bp = pd.crosstab(df1['Smoking'], df1['Blood_Pressure_Abnormality'])
plt.figure(figsize=(10, 6))
smoking_bp.plot(kind='bar', stacked=True)
plt.title('Blood Pressure Abnormality by Smoking Status')
plt.xlabel('Smoking (0=No, 1=Yes)')
plt.ylabel('Count')
plt.legend(['Normal', 'Abnormal'])
save_plot('bp_by_smoking')
print_tabulated(smoking_bp, 
                headers=['Normal BP', 'Abnormal BP'],
                title='Smoking vs Blood Pressure')

# Additional smoking analysis
smoking_stats = df1.groupby('Smoking').agg({
    'BMI': 'mean',
    'Age': 'mean',
    'Level_of_Hemoglobin': 'mean'
}).round(2)
print_tabulated(smoking_stats, headers='keys', title='Health Metrics by Smoking Status')

# Stress Levels
stress_distribution = df1['Level_of_Stress'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
sns.countplot(data=df1, x='Level_of_Stress')
plt.title('Distribution of Stress Levels')
plt.xlabel('Stress Level (1=Low, 2=Normal, 3=High)')
plt.ylabel('Count')
save_plot('stress_levels')
print_tabulated(stress_distribution.reset_index(), 
                headers=['Stress Level', 'Count'],
                title='Stress Level Distribution')

# Stress and BP relationship
stress_bp = pd.crosstab(df1['Level_of_Stress'], df1['Blood_Pressure_Abnormality'])
print_tabulated(stress_bp, 
                headers=['Normal BP', 'Abnormal BP'],
                title='Stress Levels vs Blood Pressure')

# 6. Physical Activity Analysis
avg_steps = df2.groupby('Patient_Number')['Physical_activity'].mean().reset_index()
step_stats = avg_steps['Physical_activity'].describe()
plt.figure(figsize=(10, 6))
sns.histplot(data=avg_steps, x='Physical_activity', bins=30, kde=True)
plt.title('Distribution of Average Daily Steps')
plt.xlabel('Average Steps per Day')
plt.ylabel('Count')
save_plot('steps_distribution')
print_tabulated(step_stats.to_frame(), 
                headers=['Statistic', 'Value'],
                title='Daily Steps Statistics')

# 7. Combined Analysis
combined_df = pd.merge(df1, avg_steps, on='Patient_Number', how='inner')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x='BMI', y='Physical_activity', 
                hue='Blood_Pressure_Abnormality', alpha=0.6)
plt.title('BMI vs Physical Activity')
plt.xlabel('BMI')
plt.ylabel('Average Daily Steps')
save_plot('bmi_vs_steps')

# Calculate correlation between BMI and physical activity
bmi_steps_corr = combined_df['BMI'].corr(combined_df['Physical_activity'])
print("\n=== BMI and Physical Activity Correlation ===")
print(f"Correlation coefficient: {bmi_steps_corr:.4f}")

# Activity level categories
combined_df['Activity_Level'] = pd.qcut(combined_df['Physical_activity'], 
                                      q=4, 
                                      labels=['Sedentary', 'Lightly Active', 
                                             'Moderately Active', 'Very Active'])
activity_bp = pd.crosstab(combined_df['Activity_Level'], 
                         combined_df['Blood_Pressure_Abnormality'])
print_tabulated(activity_bp, 
                headers=['Normal BP', 'Abnormal BP'],
                title='Physical Activity Levels vs Blood Pressure')

# 8. Risk Factor Analysis
# Create composite risk score
df1['risk_score'] = (
    (df1['BMI'] > 30).astype(int) +
    df1['Smoking'] +
    (df1['Level_of_Stress'] == 3).astype(int) +
    (df1['alcohol_consumption_per_day'] > df1['alcohol_consumption_per_day'].mean()).astype(int)
)

plt.figure(figsize=(10, 6))
sns.boxplot(x='risk_score', y='Level_of_Hemoglobin', data=df1)
plt.title('Hemoglobin Levels by Risk Score')
plt.xlabel('Risk Score (0-4)')
plt.ylabel('Level of Hemoglobin')
save_plot('risk_score_hemoglobin')

risk_stats = df1.groupby('risk_score').agg({
    'Blood_Pressure_Abnormality': 'mean',
    'Level_of_Hemoglobin': ['mean', 'std'],
    'Patient_Number': 'count'
}).round(3)
print_tabulated(risk_stats, 
                headers=['BP Abnormality Rate', 'Avg Hemoglobin', 'Std Hemoglobin', 'Count'],
                title='Risk Score Analysis')

# 9. Additional Health Indicators
# Analyze relationship between salt intake and blood pressure
salt_bp_corr = stats.pointbiserialr(df1['Blood_Pressure_Abnormality'], 
                                   df1['salt_content_in_the_diet'])
print("\n=== Salt Intake and Blood Pressure Correlation ===")
print(f"Correlation coefficient: {salt_bp_corr.correlation:.4f}")
print(f"p-value: {salt_bp_corr.pvalue:.4f}")

# Alcohol consumption analysis
alcohol_stats = df1.groupby('Blood_Pressure_Abnormality')['alcohol_consumption_per_day'].describe()
print_tabulated(alcohol_stats, 
                headers='keys',
                title='Alcohol Consumption by BP Status')

# 10. Summary Statistics
print("\n=== Key Health Indicators ===")
key_stats = {
    'Average BMI': df1['BMI'].mean(),
    'BP Abnormality Rate': df1['Blood_Pressure_Abnormality'].mean() * 100,
    'Average Daily Steps': avg_steps['Physical_activity'].mean(),
    'Smoking Rate': df1['Smoking'].mean() * 100,
    'High Stress Rate': (df1['Level_of_Stress'] == 3).mean() * 100,
    'Average Salt Intake': df1['salt_content_in_the_diet'].mean(),
    'Average Alcohol Consumption': df1['alcohol_consumption_per_day'].mean()
}
print_tabulated(pd.Series(key_stats).round(2).to_frame(), 
                headers=['Metric', 'Value'])

# 11. Additional Statistical Tests
# Test for normality of key variables
print("\n=== Normality Tests ===")
for variable in ['BMI', 'Level_of_Hemoglobin', 'Age']:
    stat, p_value = stats.normaltest(df1[variable])
    print(f"{variable} normality test:")
    print(f"statistic: {stat:.4f}, p-value: {p_value:.4f}")

# ANOVA test for stress levels and hemoglobin
f_stat, p_val = stats.f_oneway(*[group['Level_of_Hemoglobin'].values 
                                for name, group in df1.groupby('Level_of_Stress')])
print("\n=== ANOVA Test: Stress Levels vs Hemoglobin ===")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_val:.4f}")

# Multiple regression for blood pressure prediction
# First, handle missing values
X = df1[['Age', 'BMI', 'Level_of_Hemoglobin', 'salt_content_in_the_diet', 
         'alcohol_consumption_per_day', 'Smoking']]
y = df1['Blood_Pressure_Abnormality']

# Create and fit the imputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Fit the model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

# Print feature importance
print("\n=== Blood Pressure Predictors Importance ===")
predictors_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False)
print_tabulated(predictors_importance, headers='keys')
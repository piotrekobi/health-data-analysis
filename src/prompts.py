from langchain.prompts import PromptTemplate

class HealthQueryPrompts:
    DATABASE_SCHEMA = """Database Schema:

health:
- Patient_Number: Unique identifier
- Blood_Pressure_Abnormality: 0=Normal, 1=Abnormal
- Level_of_Hemoglobin: Hemoglobin in g/dl
- Genetic_Pedigree_Coefficient: 0-1 scale
- Age: Patient's age
- BMI: Body Mass Index
- Sex: 0=Male, 1=Female
- Pregnancy: 0=No, 1=Yes
- Smoking: 0=No, 1=Yes
- salt_content_in_the_diet: mg/per day
- alcohol_consumption_per_day: ml/day
- Level_of_Stress: 1=Low, 2=Normal, 3=High
- Chronic_kidney_disease: 0=No, 1=Yes
- Adrenal_and_thyroid_disorders: 0=No, 1=Yes

physical_activity:
- Patient_Number: Unique identifier (joins with Dataset 1)
- Day_Number: Day of measurement
- Physical_activity: Number of steps per day"""

    QUERY_TEMPLATE = PromptTemplate(
        input_variables=["question"],
        template=f"""You are a SQL query generator for a health database. Generate ONLY the SQL query without any explanation or additional text.


The database uses SQLite. Do not use functions that are unavailable in SQLite, like CORR.
Always use aggregation to reduce result size.

{DATABASE_SCHEMA}

Generate a SQL query to answer this question: {{question}}
SQL Query:
"""
    )

    ANALYSIS_TEMPLATE = PromptTemplate(
        input_variables=["question", "sql_query", "data"],
        template=f"""Based on the provided health data, provide ONLY a concise analysis answering the question in maximum three sentences and 80 words.

{DATABASE_SCHEMA}

Question: {{question}}
SQL Query Used: {{sql_query}}
Data:
{{data}}

Write the analysis. Finish your response with ***
Analysis:
"""
    )
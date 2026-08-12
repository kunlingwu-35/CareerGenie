import pandas as pd

df = pd.read_excel("CIP2020_SOC2018_Crosswalk.xlsx")

with open("cip_text.txt", "w", encoding="utf-8") as file:
    for _, row in df.iterrows():

        parts = []
        
        for column in df.columns:
            if pd.notna(row[column]):
                parts.append(f"{column}: {row[column]}")

        file.write(" | ".join(parts) + "\n")

print("Finished converting spreadsheet to text!")
import numpy as np
import pandas as pd
import sqlite3

# Create a connection and create the db
conn = sqlite3.connect('breast_cancer.db')

# Create the cursor for use
c = conn.cursor()

# Create table ipf
c.execute('''CREATE TABLE cancer (id integer,
                                    diagnosis text,
                                    radius_mean real,
                                    texture_mean real,
                                    perimeter_mean real,
                                    area_mean real,
                                    smoothness_mean real,
                                    compactness_mean real,
                                    concavity_mean real,
                                    concave_points_mean real,
                                    symmetry_mean real,
                                    fractal_dimension_mean real,
                                    radius_se real,
                                    texture_se real,
                                    perimeter_se real,
                                    area_se real,
                                    smoothness_se real,
                                    compactness_se real,
                                    concavity_se real,
                                    concave_points_se real,
                                    symmetry_se real,
                                    fractal_dimension_se real,
                                    radius_worst real,
                                    texture_worst real,
                                    perimeter_worst real,
                                    area_worst real,
                                    smoothness_worst real,
                                    compactness_worst real,
                                    concavity_worst real,
                                    concave_points_worst real,
                                    symmetry_worst real,
                                    fractal_dimension_worst real)''')

# Insert data into ipf
for i in np.array(pd.read_csv('data.csv')):
    sql_insert = '''INSERT INTO cancer VALUES ({},'{}',{},{},{},
                                               {},{},{},{},{},
                                               {},{},{},{},{},
                                               {},{},{},{},{},
                                               {},{},{},{},{},
                                               {},{},{},{},{},
                                               {},{})'''.format(i[0],i[1],i[2],i[3],i[4],
                                                                i[5],i[6],i[7],i[8],i[9],
                                                                i[10],i[11],i[12],i[13],i[14],
                                                                i[15],i[16],i[17],i[18],i[19],
                                                                i[20],i[21],i[22],i[23],i[24],
                                                                i[25],i[26],i[27],i[28],i[29],
                                                                i[30],i[31])
    c.execute(sql_insert)

# Save (commit) the changes
conn.commit()

# Close the connection to sql db
# Just be sure any changes have been committed or they will be lost.
conn.close()

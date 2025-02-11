import polars as pl
import numpy as np


runsled = 0
first = 0

def findrun(value:int):
    global runsled
    
    idx = (np.abs(runsled - value)).argmin()
    return runsled[idx]
    
    
    
df = pl.read_csv('./list_of_beam_combo.csv')

df = df.filter( ~pl.col('Run').is_null())

dfled = pl.read_csv('./list_of_purity_combo.csv')
dfled = dfled.filter(~pl.col('Run').is_null())
dfled = dfled.filter( pl.col('type') == "LEDG" ).sort('Run')
runsled = dfled['Run'].to_numpy()

dfcosmics = df.filter(
    pl.col('type') == "COSMIC" # not good, because many do not have cosmics
).sort('Run')


dfcosmics = dfcosmics.with_columns(
    pl.lit(180).alias('Efield'),
).with_columns(
    Run = pl.col('Run').cast(pl.Int16),
).with_columns(
    pl.col('Run').map_elements(findrun, return_dtype=pl.Int64).alias("Run LED")
)

dfcosmics:pl.DataFrame = dfcosmics.with_columns(
    pl.col('Time Start').str.replace_all('\.','/').str.split_exact(" ", 1).alias('Date')
).unnest('Date').with_columns(
    pl.col("field_0").alias("Date"),
    pl.col("field_1").alias("Time Start"),
).select(['Run', 'Date', 'Time Start','Efield', 'Run LED'])

with pl.Config(tbl_rows=35):
    print(dfcosmics)

dfcosmics.write_csv("beam_runs.csv")





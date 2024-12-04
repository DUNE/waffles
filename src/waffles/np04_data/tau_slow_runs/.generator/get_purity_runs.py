import polars as pl
import numpy as np


runsled = 0

def findrun(value:int):
    global runsled
    for ref in runsled:
        if ref < value <= ref+5:
            return ref
    return pl.Null
    
    
    
df = pl.read_csv('./list_of_purity_combo.csv')

df = df.filter(~pl.col('Run').is_null())

dfcosmics = df.filter( pl.col('type') == "COSMIC" ).sort('Run')
dfled = df.filter( pl.col('type') == "LEDG" ).sort('Run')
runsled = dfled['Run'].to_numpy()

# Gets efield information 
dfcosmics = dfcosmics.with_columns(
    pl.col('Configuration').str.extract_groups(r"([0-9][0-9]?[0-9]?) ?kV").struct["1"].cast(pl.Float64).alias('ef')
)

# create rows to make things easy
dfcosmics = dfcosmics.with_row_index()

# E field = 0 if first run
dfcosmics:pl.DataFrame = dfcosmics.with_columns(
    ef = pl.when(pl.col('index')==0).then(0).otherwise(
        pl.col('ef')
    )
).with_columns(
    pl.col('ef').forward_fill().alias("Efield"),
    pl.col('Start time').fill_null("??:??"),
).with_columns(
    Run = pl.col('Run').cast(pl.Int16),
).with_columns(
    pl.col('Run').map_elements(findrun, return_dtype=pl.Int64).alias("Run LED")
).with_columns(
    pl.col('Run LED').forward_fill().backward_fill()
).select(['Run','Date','Start time','Efield', 'Run LED'])

dfled = dfled.select(['Run','Date','Start time'])

with pl.Config(tbl_rows=35):
    print(dfcosmics)
    print(dfled)


dfcosmics.write_csv("purity_runs.csv")
dfled.write_csv("led_runs.csv")





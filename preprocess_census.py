import pandas as pandas


dataframe = pandas.read_csv('census-income.data.csv', header=None)
col1vals = {' Without pay': 1, ' Local government': 2, ' Private': 3, ' State government': 4, ' Federal government': 5,
            ' Not in universe': 0, ' Self-employed-incorporated': 6, ' Never worked': 7, ' Self-employed-not incorporated': 8}
col4vals = {' Doctorate degree(PhD EdD)': 0, ' Masters degree(MA MS MEng MEd MSW MBA)': 1, ' 1st 2nd 3rd or 4th grade': 2,
            ' Associates degree-academic program': 3, ' 11th grade': 4, ' 10th grade': 5, ' 9th grade': 6, ' Children': 7,
            ' Associates degree-occup /vocational': 8, ' 5th or 6th grade': 9, ' 7th and 8th grade': 10,
            ' Some college but no degree': 11, ' High school graduate': 12, ' Prof school degree (MD DDS DVM LLB JD)': 13,
            ' Less than 1st grade': 14, ' Bachelors degree(BA AB BS)': 15, ' 12th grade no diploma': 16}
col6vals = {' High school': 1, ' College or university': 2}
col7vals = {' Separated': 0, ' Married-A F spouse present': 1, ' Never married': 2, ' Married-spouse absent': 3,
            ' Married-civilian spouse present': 4, ' Divorced': 5, ' Widowed': 6}
col8vals = {' Utilities and sanitary services': 0, ' Private household services': 1, ' Personal services except private HH': 2,
            ' Construction': 3, ' Manufacturing-durable goods': 4, ' Education': 5, ' Forestry and fisheries': 7, ' Mining': 8,
            ' Not in universe or children': 9, ' Social services': 10, ' Entertainment': 11, ' Manufacturing-nondurable goods': 12,
            ' Transportation': 13, ' Agriculture': 14, ' Armed Forces': 15, ' Hospital services': 16, ' Medical except hospital': 17,
            ' Wholesale trade': 18, ' Retail trade': 19, ' Public administration': 20, ' Other professional services': 21,
            ' Finance insurance and real estate': 22, ' Business and repair services': 23, ' Communications': 24}
col9vals = {' Farming forestry and fishing': 2, ' Handlers equip cleaners etc ': 3, ' Technicians and related support': 4,
            ' Machine operators assmblrs & inspctrs': 5, ' Executive admin and managerial': 6, ' Professional specialty': 7,
            ' Precision production craft & repair': 8, ' Adm support including clerical': 9, ' Sales': 10,
            ' Other service': 11, ' Protective services': 12, ' Transportation and material moving': 13}
col10vals = {' Other': 0, ' Amer Indian Aleut or Eskimo': 1, ' Black': 2, ' Asian or Pacific Islander': 3, ' White': 4}
col11vals = {' Cuban': 0, ' Mexican-American': 1, ' Central or South American': 2, ' Other Spanish': 3, ' Chicano': 4,
             ' All other': 5, ' Puerto Rican': 6, ' NA': 7, ' Do not know': 8, ' Mexican (Mexicano)': 9}
col12vals = {' Male': 0, ' Female': 1}
col14vals = {' New entrant': 1, ' Job leaver': 2, ' Job loser - on layoff': 3, ' Re-entrant': 4, ' Other job loser': 5}
col15vals = {' Unemployed full-time': 0, ' Children or Armed Forces': 1, ' PT for non-econ reasons usually FT': 2,
             ' Not in labor force': 3, ' PT for econ reasons usually PT': 4, ' Unemployed part- time': 5,
             ' Full-time schedules': 6, ' PT for econ reasons usually FT': 7}
col19vals = {' Joint one under 65 & one 65+': 0, ' Single': 1, ' Joint both 65+': 2, ' Head of household': 3,
             ' Joint both under 65': 4, ' Nonfiler': 5}
col20vals = {' Abroad': 1, ' West': 2, ' Midwest': 3, ' South': 4, ' Northeast': 5}
col21vals = {' North Dakota': 1, ' Colorado': 2, ' Oregon': 3, ' South Carolina': 4, ' Kentucky': 5, ' Missouri': 6,
             ' New Jersey': 7, ' Texas': 8, ' Mississippi': 9, ' Arkansas': 10, ' District of Columbia': 11, ' Florida': 12,
             ' Kansas': 13, ' California': 14, ' Idaho': 15, ' Ohio': 16, ' Maine': 17, ' Oklahoma': 18, ' Alabama': 19,
             ' Louisiana': 20, ' New Hampshire': 21, ' Wisconsin': 22, ' Georgia': 23, ' Iowa': 24, ' West Virginia': 25,
             ' Delaware': 26, ' Vermont': 27, ' Abroad': 28, ' Michigan': 29, ' Nebraska': 30, ' New York': 31,
             ' Illinois': 32, ' Arizona': 33, ' Wyoming': 34, ' North Carolina': 35, ' Indiana': 36, ' Minnesota': 37,
             ' Alaska': 38, ' Montana': 39, ' Virginia': 40, ' Utah': 41, ' Connecticut': 42, ' South Dakota': 43,
             ' Pennsylvania': 44, ' Tennessee': 45, ' New Mexico': 46, ' Massachusetts': 47, ' Maryland': 48, ' Nevada': 49}
col22vals = {' Grandchild <18 never marr child of subfamily RP': 0, ' Child 18+ spouse of subfamily RP': 1,
             ' Child 18+ ever marr RP of subfamily': 2, ' Grandchild 18+ never marr not in subfamily': 3,
             ' Other Rel <18 never marr child of subfamily RP': 4, ' Child 18+ never marr Not in a subfamily': 5,
             ' Child 18+ never marr RP of subfamily': 6, ' Child 18+ ever marr Not in a subfamily': 7,
             ' Child <18 spouse of subfamily RP': 8, ' Secondary individual': 9, ' Other Rel 18+ ever marr RP of subfamily': 10,
             ' Child <18 ever marr not in subfamily': 11, ' Child <18 never marr RP of subfamily': 12,
             ' Grandchild <18 ever marr not in subfamily': 13, ' Other Rel <18 spouse of subfamily RP': 14, ' Nonfamily householder': 15,
             ' Other Rel 18+ ever marr not in subfamily': 16, ' Child <18 ever marr RP of subfamily': 17, ' In group quarters': 18,
             ' Child under 18 of RP of unrel subfamily': 19, ' Child <18 never marr not in subfamily': 20,
             ' Other Rel 18+ spouse of subfamily RP': 21, ' Other Rel 18+ never marr RP of subfamily': 22,
             ' Grandchild 18+ spouse of subfamily RP': 23, ' Other Rel 18+ never marr not in subfamily': 24,
             ' Grandchild <18 never marr not in subfamily': 24, ' Other Rel <18 never marr not in subfamily': 25,
             ' Grandchild 18+ never marr RP of subfamily': 26, ' Grandchild 18+ ever marr RP of subfamily': 27,
             ' Other Rel <18 never married RP of subfamily': 28, ' Grandchild 18+ ever marr not in subfamily': 29,
             ' Spouse of householder': 30, ' Other Rel <18 ever marr not in subfamily': 31,
             ' Other Rel <18 ever marr RP of subfamily': 32, ' Grandchild <18 never marr RP of subfamily': 33,
             ' RP of unrelated subfamily': 34, ' Spouse of RP of unrelated subfamily': 35, ' Householder': 36}
col23vals = {' Other relative of householder': 0, ' Group Quarters- Secondary individual': 1, ' Spouse of householder': 2,
             ' Child under 18 ever married': 3, ' Child under 18 never married': 4, ' Nonrelative of householder': 5,
             ' Child 18 or older': 6, ' Householder': 7}
col25vals = {' MSA to MSA': 0, ' NonMSA to MSA': 1, ' Not identifiable': 2, ' MSA to nonMSA': 3, ' Abroad to MSA': 4,
             ' Abroad to nonMSA': 5, ' Nonmover': 6, ' NonMSA to nonMSA': 7}
col26vals = {' Same county': 2, ' Different region': 3, ' Different division same region': 4, ' Different state same division': 8,
             ' Nonmover': 7, ' Different state in South': 10, ' Different state in Northeast': 6,
             ' Different county same state': 5, ' Different state in Midwest': 9, ' Different state in West': 8}
col28vals = {' Not in universe under 1 year old': 0, ' No': 1, ' Yes': 2}
col31vals = {' Mother only present': 1, ' Father only present': 2, ' Neither parent present': 3, ' Both parents present': 4}
col32vals = {' El-Salvador': 23, ' Outlying-U S (Guam USVI etc)': 1, ' Portugal': 2, ' Trinadad&Tobago': 3, ' Vietnam': 4,
             ' Thailand': 5, ' Ecuador': 6, ' Poland': 7, ' Columbia': 8, ' Yugoslavia': 9, ' Laos': 10, ' Canada': 11,
             ' Japan': 12, ' Nicaragua': 13, ' France': 14, ' China': 15, ' Cuba': 16, ' Jamaica': 17, ' Hungary': 18,
             ' Panama': 19, ' United-States': 20, ' Philippines': 21, ' Haiti': 22, ' ?': 0, ' Cambodia': 24, ' Greece': 25,
             ' Scotland': 26, ' Ireland': 27, ' Peru': 28, ' Hong Kong': 29, ' Taiwan': 30, ' Honduras': 31, ' Germany': 32,
             ' Italy': 33, ' South Korea': 34, ' India': 35, ' Mexico': 36, ' Iran': 37, ' Puerto-Rico': 38, ' Dominican-Republic': 39,
             ' Holand-Netherlands': 40, ' Guatemala': 41, ' England': 42}
col35vals = {' Native- Born in Puerto Rico or U S Outlying': 0, ' Native- Born in the United States': 1,
             ' Foreign born- Not a citizen of U S ': 2, ' Foreign born- U S citizen by naturalization': 3,
             ' Native- Born abroad of American Parent(s)': 4}
col41vals = {' - 50000.': 0, ' 50000+.': 1}
dataframe.replace(col1vals, inplace=True)
dataframe.replace(col4vals, inplace=True)
dataframe.replace(col6vals, inplace=True)
dataframe.replace(col7vals, inplace=True)
dataframe.replace(col8vals, inplace=True)
dataframe.replace(col9vals, inplace=True)
dataframe.replace(col10vals, inplace=True)
dataframe.replace(col11vals, inplace=True)
dataframe.replace(col12vals, inplace=True)
dataframe.replace(col14vals, inplace=True)
dataframe.replace(col15vals, inplace=True)
dataframe.replace(col19vals, inplace=True)
dataframe.replace(col20vals, inplace=True)
dataframe.replace(col21vals, inplace=True)
dataframe.replace(col22vals, inplace=True)
dataframe.replace(col23vals, inplace=True)
dataframe.replace(col25vals, inplace=True)
dataframe.replace(col26vals, inplace=True)
dataframe.replace(col28vals, inplace=True)
dataframe.replace(col31vals, inplace=True)
dataframe.replace(col32vals, inplace=True)
dataframe.replace(col35vals, inplace=True)
dataframe.replace(col41vals, inplace=True)
for i in range(0, 42):
    print(i, set(dataframe[i].tolist()))
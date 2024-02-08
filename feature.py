def umur(x):
    if x <= 12: #Anak-anak
        return "1"
    elif 13 <= x <= 18: #Remaja
        return "2"
    elif 19 <= x <= 40: #Dewasa
        return "3"
    elif 41 <= x <= 65: #Orang Tua
        return "4"
    else:
        return "5"

def survival_status(x):
    if x['Pclass'] == 1 and x['Sex'] == 'male': #Berada pada kelas "pertama" AND merupakan "laki-laki" = 1
        return 1  # Survived
    elif x['Pclass'] == 2 and x['Age'] <= 12: #Berada pada kelas "kedua" AND usianya Anak-anak
        return 1  # Survived
    else:
        return 0  # Not Survived

def new_feature(df):
    df['umur'] = df['Age'].apply(lambda x: umur(x))
    df['WA'] = 0
    df.loc[(df['umur'] == '1') & (df['Sex'] == 'female'), 'WA'] = 1
    df['MA'] = 0
    df.loc[(df['umur'] == '1') & (df['Sex'] == 'male'), 'MA'] = 1
    df['Survival'] = df.apply(survival_status, axis=1)
    return df

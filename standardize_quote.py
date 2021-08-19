
# standardize the quotation mark, replace '’'(ascii 8217) with "'" (ascii 39)
def standardize_quote(str):
    temp = str
    for i in range(0, len(str) - 1):
        if ord(str[i]) == 8217:
            temp = temp[:i] + "'" + temp[i+1:]
    return temp
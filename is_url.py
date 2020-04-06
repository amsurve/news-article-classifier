import re

def is_url(text):
    regex = re.compile(
            r'^(?:http|ftp)s?://' # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
            r'localhost|' #localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
            r'(?::\d+)?' # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return(re.match(regex, text) is not None)            

if __name__ == "__main__":
    print(is_url("https://stackoverflow.com/questions/7160737/python-how-to-validate-a-url-in-python-malformed-or-not"))
    print(is_url("https://stackoverflow.com sdksd"))
    print(is_url("skd ashttps://stackoverflow.com sdksd"))
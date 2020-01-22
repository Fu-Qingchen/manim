from aip import AipSpeech

APP_ID = '17501764'
API_KEY = 'oXzyx5nxmF6whGL2TEKW6MoG'
SECRET_KEY = 'zK79FAoMLzIgKer1GrMYDjIUXnUGkPnI'

def tts(str, id):
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)
    result = client.synthesis(str, 'zh', 1, {'spd': 6, 'vol': 9, 'per': 3})
    filename = 'result/audio'
    if not isinstance(result, dict):
        with open(filename + '{}'.format(id) + '.mp3', 'wb') as f:
            f.write(result)

text = []

if __name__ == '__main__':
    file = open("text.txt")
    while True:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:
            text.append(line)
    for i in range(len(text)):
        tts(text[i], i+1)
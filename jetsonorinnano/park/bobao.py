from pydub import AudioSegment
from pydub.playback import play


audio = AudioSegment.from_wav('/home/wheeltec/park/src/park/music/potato.wav')
play(audio)


import traceback
import json
from oqtopus_sse_pulse.libs.miyanaga.myfunc import hello

print("start program")
try:
    result :dict = {}
    result["message"] = hello()
    print("payload="+  json.dumps(result, ensure_ascii=False, separators=(",", ":")))

except Exception as e:
    print("Exception:", e)
    traceback.print_exc()
finally:
    print("end program")

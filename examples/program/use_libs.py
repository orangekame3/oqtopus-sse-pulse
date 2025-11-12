import traceback

from oqtopus_sse_pulse.libs.test_utils import hello

print("start program")
try:
    hello()
    print("payload=")

except Exception as e:
    print("Exception:", e)
    traceback.print_exc()
finally:
    print("end program")

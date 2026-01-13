import requests
import time
import json
from metrics import results_quality

with open('queries_train.json', 'rt') as f:
  queries = json.load(f)

# queries = {"Silk Road trade cultural exchange": []}

qs_res = []
EXTERNAL="34.170.95.149"
for q, true_wids in queries.items():
    duration, ap = None, None
    t_start = time.time()
    try:
        # res = requests.get(url + '/search', {'query': q}, timeout=35)
        response = requests.get(f"http://{EXTERNAL}:8080/search?query={q}", timeout=35)
        duration = time.time() - t_start

        if response.status_code == 200:
            pred_wids, _ = zip(*response.json())
            rq = results_quality(true_wids, pred_wids)

            qs_res.append((q, duration, rq))
            print(f"Query: {q}\nDuration: {duration:.2f}s, Results Quality: {rq}\n")
        else:
            print(f"Failed to get results for query: {q}, Status Code: {response.status_code}")

    except Exception as e:
        print(f"Error during request: {e}")
        continue

average_time = sum(d for _, d, _ in qs_res) / (len(qs_res) + 1)
average_rq = sum(rq for _, _, rq in qs_res) / (len(qs_res) + 1)
print(f"Average Time per Query: {average_time:.2f}s")
print(f"Average Results Quality: {average_rq}")

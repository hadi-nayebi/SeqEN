import requests


class DataPipeline:
    """methods and tools to fetch data from online resources"""

    def __init__(self):
        self.seq_url = "https://www.ebi.ac.uk/proteins/api/uniparc/sequence"

    def fetch_by_seq(self, seq):
        headers = {"Content-Type": "text/plain", "Accept": "application/json"}
        r = requests.post(self.seq_url, headers=headers, data=seq)
        if not r.ok:
            r.raise_for_status()
        return r.text

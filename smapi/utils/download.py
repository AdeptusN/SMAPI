import functools
import pathlib
import shutil
import requests
from tqdm.auto import tqdm

def download(url, file):

    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f'Request to {url} returned status code {r.status_code}')
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return True

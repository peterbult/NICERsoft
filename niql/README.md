
# niql

## Install

To run niql a couple of python packages must be installed. First install the pip dependencies

```shell
pip install -r requirements.txt
```

Then install `nicerlab` as

```shell
git clone http://github.com/peterbult/nicerlab.git
cd nicerlab
make install
```

## Usage

To use niql type, from the project root

```shell
python niql/app.py --obsdir [/path/to/obsid]
```

This will analyze the ObsID and launch the webapp server locally. You can find niql in browser at
```
http://127.0.0.1:8050/
```

**warning**: niql is in development and not guaranteed to be fool-proof. Check back for updates.

**warning**: some observation files require a lot of memory to process, causing niql to become slow or possibly unresponsive. Further optimizations is needed to address this issue. 

## Feedback

Found an issue? Please let me know!


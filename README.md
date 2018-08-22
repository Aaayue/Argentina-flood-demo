```
This is either madness or brilliance.It's remarkable how often those two traits coincide.

                                        --Pirates of the Caribbean: Dead Men Tell No Tales

```
![alttext](https://s1.1zoom.ru/big0/232/Assassin's_Creed_4_Black_506963.jpg)
searover Version 0.1


searover include three step: SentinelSLC, WaterExtract, Copy and clean local path.

- [x] SentinelSLC:
    - [x] Function:
        - [x] Step 1: check process data status
        - [x] Step 2: unzip raw  data
        - [x] Step 3: create process xml
        - [x] Step 4: gpt process
    - [x] output:
        - [x] process_status
            - [x] 0: success
            - [x] 1: data has been processed
            - [x] 2: unzip data has problem
            - [x] 3: create xml failed
            - [x] 4: gpt failed
- [x] WaterExtract:
    - [x] Function:
        - [x] covert to img to bit by threshold
        - [x] input:
            - [x] img: ...
        - [x] output:
            - [x] (img, int): ..
            - [x] int: threshold

- [x] copy_clean_local:
    - [x] Function:
        - [x] deleted the file, then copy file to result path
        - [x] input:
            - [x] local_path: local_path from SentinelSLC
            - [x] result_root: result to put,such as 'tq-data05'
            - [x] path_pattern: setting para
            - [x] suffix_pattern: temp file list
        - [x] output:
            - [x] None, False: process failure
            - [x] str, True: return the path
            

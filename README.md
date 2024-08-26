## 한국어 LLM Langserve + RAG API Example

### Codes from

- [Teddynote](https://github.com/teddylee777/langserve_ollama). 여기에 RAG를 적용한 Langchain을 호출할 수 있게 API만 추
  가한 예제입니다!

### How to use

1. git clone this repository.
2. install all requirements
   ```bash
   pip install -r requirements
   ```

- install ollama <br/> `Linux`: `curl -fsSL https://ollama.com/install.sh | sh` <br/> `others`:
  [Download Link](https://ollama.com/download/mac)

- download LLM model to `ollama-modelfile/EEVE-Korean-Instruct-10.8B-v1.0`
  ```
  huggingface-cli download \
    heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF \
    ggml-model-Q5_K_M.gguf \
    --local-dir path_to_ollama-modelfile/EEVE-Korean-Instruct-10.8B-v1.0 \   // You should replace here with proper path.
    --local-dir-use-symlinks False
  ```

3. Check Modelfile at ollama-modelfile/EEVE-Korean-Instruct-10.8B-v1.0/Modelfile whether look like following or not.

   ```
   FROM ggml-model-Q5_K_M.gguf

   TEMPLATE """{{- if .System }}
   <s>{{ .System }}</s>
   {{- end }}
   <s>Human:
   {{ .Prompt }}</s>
   <s>Assistant:
   """
   ...
   ```

4. Create model & run it

- Create LLM model
  ```
  ollama create PETER -f EEVE-Korean-Instruct-10.8B-v1.0-GGUF/Modelfile
  ```
- Run the Model.

  ```
  ollama run PETER:latest
  ```

- After moving into the `app/` directory, run the server.
  ```
  python server.py
  ```

### License

- 소스코드를 활용하실 때는 반드시 출처를 표기해 주시기 바랍니다.

  ```
  MIT License

  Copyright (c) 2024, 테디노트

  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
  ```

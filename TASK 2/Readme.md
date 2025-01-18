# Task 2: Recommendation of Conference for a Given paper

* Replace API Keys for ChatTogether in `agentic_workflow` Line Number 30. Use 2 API_Keys for fall back. The keys are rotated during execution so that Rate Limit is compensated.<br>
* Run `vectorstoreServer.py` to run the VectorStore Server by Pathway. The Server is connected to `Parsed_Docs/Parsed_Docs/Reference` folder with dynamic indexing. Any more PDFs can be added to it.<br>
* For evaluating a pdf file, put the pdf in `input` folder, replace the API Keys and run `main.py`. The output will be stored as `results.csv`.

<img width="911" alt="Screenshot 2025-01-18 at 3 37 24 PM (1)" src="https://github.com/user-attachments/assets/9ddd538c-8635-4eef-9c8d-844e6e7425d2" />

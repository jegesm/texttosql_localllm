How to build and install pakage:
* python setup.py sdist bdist_wheel
* pip install .


## Possible enhancements
* Needs to be loading metadata from the database and/or separate document db
* examples loaded from db as well
streamlit run 02-app.py --server.baseUrlPath=$REPORT_URL --server.port=9000 --server.address=0.0.0.0 --server.runOnSave=True --server.allowRunOnSave=True

## SQL validation
* https://www.promptfoo.dev/docs/guides/text-to-sql-evaluation/
* https://arize.com/blog/text-to-sql-evaluating-sql-generation-with-llm-as-a-judge/

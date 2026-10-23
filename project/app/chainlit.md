# Ergobot: The [Ergo-log](https://www.ergo-log.com/) chatbot
[Ergo-log](https://www.ergo-log.com/) defines their site as:

"*Ergo-Log is a news site about legal and illegal ergogenic aids. When they work, ergogenic aids make us stronger, faster, cleverer and younger. They often have side effects, some of which we can live with, others not.*"

## Why Ergobot?
As a fan of [ergo-log](https://www.ergo-log.com/) and [ergogenics](https://www.ergogenics.org/index.html) I always wanted to be able to just ask questions related to their content

## How does it work?
I have scraped the archive of [Ergo-log](https://www.ergo-log.com/) (2008–present) and created a database. The database is refreshed by re-scraping the archive. When asking Ergobot a question the following steps are taken:
1. An open model (qwen3.8-27b), powered by [ETHUX](https://ethux.net) on their own GPUs in the Netherlands, searches the archived articles via a retrieval tool.
2. The model answers using only the returned articles (including their URLs).
3. The answer is shown in the chat window, together with links to the articles it was based on.

---
## How to use Ergobot
[Ergo-log](https://www.ergo-log.com/) has numerous articles related to these ergogenic aids, some example questions are:
- What are the health benefits of green tea?
- Why do athletes eat bananas?
- Is ginger healthy? Why?
- How to gain muscle mass?
- What substances enhance endurance for athletes and how?

---
### Details:
- Code: [Github](https://github.com/ROpdam/ergo_bot)
- Author: Robin Opdam
- Contact: [LinkedIn](https://www.linkedin.com/in/robinopdam/)
- More projects: https://ropdam.github.io/

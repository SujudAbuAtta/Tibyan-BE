const puppeteer = require("puppeteer");

let startTime = Date.now();

const createCsvWriter = require("csv-writer").createObjectCsvWriter;
const csvWriter = createCsvWriter({
  path: "articles-" + startTime + ".csv",
  header: [
    { id: "label", title: "LABEL" },
    { id: "title", title: "TITLE" },
    { id: "url", title: "URL" },
    { id: "imgSrc", title: "IMAGE" },
  ],
});

const categories = [
  {
    title: "خادع",
    baseUrl: "https://fatabyyano.net/newsface/deceptive/page/",
  },
  {
    title: "رأي",
    baseUrl: "https://fatabyyano.net/newsface/opinion/page/",
  },
  {
    title: "زائف",
    baseUrl: "https://fatabyyano.net/newsface/fake/page/",
  },
  {
    title: "زائف جزئياً",
    baseUrl: "https://fatabyyano.net/newsface/partially-fake/page/",
  },
  {
    title: "ساخر",
    baseUrl: "https://fatabyyano.net/newsface/sarcastic/page/",
  },
  {
    title: "صحيح",
    baseUrl: "https://fatabyyano.net/newsface/correct/page/",
  },
  {
    title: "عنوان مضلل",
    baseUrl: "https://fatabyyano.net/newsface/misleading-title/page/",
  },
  {
    title: "غير مؤهل",
    baseUrl: "https://fatabyyano.net/newsface/not-qualified/page/",
  },
  {
    title: "محتوى ناقص",
    baseUrl:
      "https://fatabyyano.net/newsface/%d9%85%d8%ad%d8%aa%d9%88%d9%89-%d9%86%d8%a7%d9%82%d8%b5/page/",
  },
  {
    title: "مضلل",
    baseUrl: "https://fatabyyano.net/newsface/misleading/page/",
  },
];

var browser;
var page;

async function savePageArticles(pageArticles) {
  return csvWriter.writeRecords(pageArticles);
}

async function getCategoryArticles(cat) {
  console.log("----------------------------------------------------------");
  console.log(cat.title);
  console.log("----------------------------------------------------------");

  return new Promise(async (resolve, reject) => {
    let pageNum = 1;
    let articlesCount = 0;
    let labeledCategoryArticles = [];

    do {
      let pageArticles = await getPageArticles(
        cat.baseUrl + pageNum,
        cat.title
      );

      articlesCount = pageArticles.length;

      if (articlesCount > 0) {
        labeledPageArticles = labelArticles(pageArticles, cat.title);

        await savePageArticles(labeledPageArticles);
        labeledCategoryArticles.push(...labeledPageArticles);
        pageNum++;
      }
    } while (articlesCount > 0);

    return resolve(labeledCategoryArticles);
  });
}

function labelArticles(articles, label) {
  articles.forEach(function (article) {
    article.label = label;
  });

  return articles;
}

async function getPageArticles(pageUrl) {
  console.log("Fetching: " + pageUrl);

  return new Promise(async (resolve, reject) => {
    await page.goto(pageUrl).catch((error) => {
      console.log("XXXXXXXXXXXXXXX  ERROR  " + pageUrl);
    });

    let urls = await page
      .evaluate(() => {
        let results = [];
        let items = document.querySelectorAll("article.post");

        items.forEach((item) => {
          let img = item.querySelector("img");
          let title = item.querySelector(".post_title>a");

          if (!title) {
            return;
          }

          results.push({
            url: title.getAttribute("href"),
            title: title.innerText,
            imgSrc: img ? img.getAttribute("src") : null,
          });
        });

        return results;
      })
      .catch((error) => {
        console.log("XXXXXXXXXXXXXXX  ERROR  " + pageUrl);
      });

    return resolve(urls);
  });
}

async function start() {
  console.log("بسم الله الرحمن الرحيم");

  browser = await puppeteer.launch();
  page = await browser.newPage();

  let allArticles = [];

  for (const cat of categories) {
    await getCategoryArticles(cat);
  }

  return new Promise(async (resolve, reject) => {
    resolve(true);
  });
}

start().then(function () {
  let totalRunTime = Date.now() - startTime;

  browser.close();
  console.log("DONE!");
  console.log("----------------------------------------------------------");
  console.log("----------------------------------------------------------");
  console.log("Total runtime: " + totalRunTime / 1000 + " seconds");
});
a;

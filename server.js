const url = require('node:url');
const http = require("http");
const PORT = process.env.PORT || 8080;
const posts = require("./data");
const get = require("./get");
const post = require("./post");
const put = require("./put");
const deleteR = require("./delete");


const user = {
  id: 1, name: "name1", last: 'last1',
  id: 2, name: "name2", last: 'last2',
  id: 3, name: "name3", last: 'last3',
  id: 4, name: "name4", last: 'last4',
  id: 5, name: "name5", last: 'last5',
}
const product = {
  pid: 1, proname: "pr1",
  pid: 2, proname: "pr2",
  pid: 3, proname: "pr3",
  pid: 4, proname: "pr4",
  pid: 5, proname: "pr5",
}



const server = http.createServer((request, response) => {
  request.posts = posts;

  switch (request.method) {
    case "GET":
      response.statusCode = 400;
      response.write(`CANNOT GET ${request.url}`);
      response.end();
      break;

    case "POST":
      // response.statusCode = 400;
      const myURL = new URL('https://localhost:8080' + request.url);
      console.log(myURL.href);
      console.log(myURL.pathname)
      console.log(myURL.search);
      response.write(`CANNOT PUT ${request.url}`);
      response.end();
      break;

    case "PUT":
      response.statusCode = 400;
      response.write(`CANNOT PUT ${request.url}`);
      response.end();
      break;

    case "DELETE":
      response.statusCode = 400;
      response.write(`CANNOT DELETE ${request.url}`);
      response.end();
      break;

    default:
      response.statusCode = 400;
      response.write("No Response");
      response.end();
  }
});

server.listen(PORT, (err) => {
  err ? console.error(err) : console.log(`listening on port ${PORT}`);
});

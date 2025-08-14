
import { ConfigProvider } from '@douyinfe/semi-ui';
import enUS from '@douyinfe/semi-ui/lib/es/locale/source/en_US';

import { createRoot } from "react-dom/client";

import { BrowserRouter } from "react-router-dom";
import { Routes, Route } from "react-router-dom";
import About from "./pages/About";
import { NoMatch } from "./pages/NoMatch";
import BackendLayout from "./pages/Layout/Backend";
import NodePage from "./pages/Node";
import Design from "./pages/Layout/Design/Design";
import './assets/css/index.scss'
//import './mock'
import Home from "./pages/Home";

const app = createRoot(document.getElementById("root")!);

app.render(
  <ConfigProvider locale={enUS}>

    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/design" element={<Design />} />



        <Route path="/backend" element={<BackendLayout />} >

        </Route>

        <Route path="/backendItem" >
          <Route path="node" element={<NodePage />} />
        </Route>
        <Route path="about" element={<About />} />



        <Route path="*" element={<NoMatch />} />
      </Routes>
    </BrowserRouter>
  </ConfigProvider>
);

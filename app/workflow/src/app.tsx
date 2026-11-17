
import { ConfigProvider } from '@douyinfe/semi-ui';
import enUS from '@douyinfe/semi-ui/lib/es/locale/source/en_US';

import { createRoot } from "react-dom/client";

import { BrowserRouter } from "react-router-dom";
import { Routes, Route } from "react-router-dom";
import About from "./pages/About";
import { NoMatch } from "./pages/NoMatch";
import BackendLayout from "./pages/Layout/Backend";
import NodePage from "./pages/Node";
import EditorPages from './pages/EditorPages';
import './assets/css/index.scss'
//import './mock'
import Home from "./pages/Home";
import { EditorAccess } from './components/editor-access';

const app = createRoot(document.getElementById("root")!);

app.render(
  <ConfigProvider locale={enUS}>

    <EditorAccess><BrowserRouter>
      <Routes>
        <Route element={<EditorPages />}>
          <Route path="/" element={<Home />} />
          <Route path="/design" element={null} />
        </Route>



        <Route path="/backend" element={<BackendLayout />} >

        </Route>

        <Route path="/backendItem" >
          <Route path="node" element={<NodePage />} />
        </Route>
        <Route path="about" element={<About />} />



        <Route path="*" element={<NoMatch />} />
      </Routes>
    </BrowserRouter></EditorAccess>
  </ConfigProvider>
);

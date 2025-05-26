import { createRoot } from "react-dom/client";

import { Flow } from "./pages/components/flow/Index";

import { BrowserRouter } from "react-router-dom";
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import About from "./pages/About";
import { NoMatch } from "./pages/NoMatch";
import HomeLayout from "./pages/Layout/Home";
import Flows from "./pages/flows";
import Backend from "./pages/Layout/Backend/Backend";
import './mock'

const app = createRoot(document.getElementById("root")!);

app.render(
  <BrowserRouter>
    <Routes>
      <Route path="/backend" element={<Backend />} />

      <Route path="flows" element={<Flows />} />

      <Route index element={<Home />} />
      <Route path="about" element={<About />} />

      <Route path="*" element={<NoMatch />} />
    </Routes>
  </BrowserRouter>
);

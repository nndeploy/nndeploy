import request from "./request";
import { IBusinessNode } from "./pages/Layout/Design/WorkFlow/entity";
import { IWorkFlowShortEntity } from "./entity";

// const images: string[] = [
//   'https://images.media.io/images2025/index/auto-subtitles1.png', 
//   'https://images.media.io/images2025/index/video-enhancer.png', 
//   'https://images.media.io/images2025/index/text-to-music.png',

//   'https://images.media.io/images2025/index/auto-subtitles1.png', 
//   'https://images.media.io/images2025/index/video-enhancer.png', 
//   'https://images.media.io/images2025/index/text-to-music.png',
// ]

export async function apiGeTemplates() {
  var response = await request.get<IWorkFlowShortEntity[]>(
    "/api/template",
    {}
  );

  return response;
}


export async function apiGetWorkflows() {
  var response = await request.get<IWorkFlowShortEntity[]>(
    "/api/workflows",
    {}
  );

  return response;
}


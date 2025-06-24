import Mock, { MockjsMock, templateOrFn } from "mockjs";
import {
  IResourceBranchEntity,
  IResourceEntity,
  IResourceTreeNodeEntity,
} from "../../../pages/Layout/Design/Resource/entity";
import { MockItem } from "../../entity";

// mock方法,详细的可以看官方文档
const Random = Mock.Random;

const resoures = {
  image: {
    animal: {
      cat: [
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
        "https://images.pexels.com/photos/57416/cat-sweet-kitty-animals-57416.jpeg",
        "https://images.pexels.com/photos/774731/pexels-photo-774731.jpeg",
      ],
      dog: [
        "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
        "https://images.pexels.com/photos/551628/pexels-photo-551628.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/3687770/pexels-photo-3687770.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/532310/pexels-photo-532310.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
      ],
    },

    people: {
      man: [
        "https://images.pexels.com/photos/874158/pexels-photo-874158.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/845434/pexels-photo-845434.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/1036627/pexels-photo-1036627.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/977796/pexels-photo-977796.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
      ],
      woman: [
        "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/1758144/pexels-photo-1758144.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
        "https://images.pexels.com/photos/774095/pexels-photo-774095.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
      ],
    },
  },
  video: {
    animal: {
      cat: [
        "https://videos.pexels.com/video-files/1481903/1481903-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/3116737/3116737-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/854532/854532-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/3009091/3009091-sd_640_360_30fps.mp4",
      ],
      dog: [
        "https://videos.pexels.com/video-files/3042473/3042473-sd_640_360_30fps.mp4",
        "https://videos.pexels.com/video-files/853770/853770-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/1851002/1851002-sd_640_360_24fps.mp4",
        "https://videos.pexels.com/video-files/4057316/4057316-sd_506_960_25fps.mp4",
      ],
    },
    person: {
      man: [
        "https://videos.pexels.com/video-files/3209176/3209176-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/3126361/3126361-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/2795749/2795749-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/3202048/3202048-sd_640_360_25fps.mp4",
      ],
      woman: [
        "https://videos.pexels.com/video-files/2795406/2795406-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/3048162/3048162-sd_640_360_24fps.mp4",
        "https://videos.pexels.com/video-files/1122526/1122526-sd_640_360_25fps.mp4",
        "https://videos.pexels.com/video-files/3048880/3048880-sd_640_360_24fps.mp4",
      ],
    },
  },
};

const branchData: IResourceBranchEntity[] = [
  {
    id: "image",
    parentId: "",
    name: "image",
  },
  // {
  //   id: "image-animal",
  //   parentId: "image",
  //   name: "animal",
  // },
  // {
  //   id: "image-cat",
  //   parentId: "image-animal",
  //   name: "image",
  // },

  // {
  //   id: "image-dog",
  //   parentId: "image-animal",
  //   name: "dog",
  // },

  // {
  //   id: "image-person",
  //   parentId: "image",
  //   name: "person",
  // },
  // {
  //   id: "image-man",
  //   parentId: "image-person",
  //   name: "man",
  // },

  // {
  //   id: "image-woman",
  //   name: "image-woman",
  //   parentId: "image-person",
  // },

  {
    id: "video",
    parentId: "",
    name: "video",
  },
  // {
  //   id: "video-animal",
  //   parentId: "video",
  //   name: "animal",
  // },
  // {
  //   id: "video-cat",
  //   parentId: "video-animal",
  //   name: "cat",
  // },

  // {
  //   id: "video-dog",
  //   parentId: "video-animal",
  //   name: "dog",
  // },

  // {
  //   id: "video-person",
  //   parentId: "video",
  //   name: "person",
  // },
  // {
  //   id: "video-man",
  //   parentId: "video-person",
  //   name: "man",
  // },

  // {
  //   id: "video-woman",
  //   parentId: "video-person",
  //   name: "woman",
  // },
];
const resourceData = [
  {
    id: "image-cat1",
    parentId: "image",
    name: "cat1",

    mime: "image",
    url: "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
  },
  {
    id: "image-cat2",
    parentId: "image",
    name: "cat2",

    mime: "image",
    url: "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
  },
  {
    id: "image-dog1",
    parentId: "image",
    name: "dog1",

    mime: "image",
    url: "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
  },
  {
    id: "image-dog2",
    parentId: "image",
    name: "dog2",

    mime: "image",
    url: "https://images.pexels.com/photos/551628/pexels-photo-551628.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
  },

  {
    id: "image-man1",
    parentId: "image",
    name: "man1",

    mime: "image",
    url: "https://images.pexels.com/photos/874158/pexels-photo-874158.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
  },
  {
    id: "image-man2",
    parentId: "image",
    name: "man2",

    mime: "image",
    url: "https://images.pexels.com/photos/845434/pexels-photo-845434.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
  },
  {
    id: "image-woman1",
    parentId: "image",
    name: "woman1",

    mime: "image",
    url: "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
  },
  {
    id: "image-woman2",
    parentId: "image",
    name: "woman2",

    mime: "image",
    url: "https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
  },

  {
    id: "video-cat1",
    parentId: "video",
    name: "cat1",

    mime: "video",
    url: "https://videos.pexels.com/video-files/1481903/1481903-sd_640_360_25fps.mp4",
  },
  {
    id: "video-cat2",
    parentId: "video",
    name: "cat2",

    mime: "video",
    url: "https://videos.pexels.com/video-files/3116737/3116737-sd_640_360_25fps.mp4",
  },
  {
    id: "video-dog1",
    parentId: "video",
    name: "dog1",

    mime: "video",
    url: "https://videos.pexels.com/video-files/3042473/3042473-sd_640_360_30fps.mp4",
  },
  {
    id: "video-dog2",
    parentId: "video",
    name: "dog2",

    mime: "video",
    url: "https://videos.pexels.com/video-files/853770/853770-sd_640_360_25fps.mp4",
  },
  {
    id: "video-man1",
    parentId: "video",
    name: "man1",

    mime: "video",
    url: "https://videos.pexels.com/video-files/3209176/3209176-sd_640_360_25fps.mp4",
  },
  {
    id: "video-man2",
    parentId: "video",
    name: "man2",

    mime: "video",
    url: "https://videos.pexels.com/video-files/3126361/3126361-sd_640_360_25fps.mp4",
  },
  {
    id: "video-woman1",
    parentId: "video",
    name: "woman1",

    mime: "video",
    url: "https://videos.pexels.com/video-files/2795406/2795406-sd_640_360_25fps.mp4",
  },
  {
    id: "video-woman2",
    parentId: "video",
    name: "woman2",

    mime: "video",
    url: "https://videos.pexels.com/video-files/3048162/3048162-sd_640_360_24fps.mp4",
  },
];

export const resourceHandler: MockItem[] = [
  {
    url: "/resource/branch/save",
    type: "post",
    response: (options) => {
      const resource: IResourceEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {
          ...resource,
          id: resource.id ? resource.id : Random.guid(),
        },
      };
    },
  },
  {
    url: "/resource/tree",
    type: "get",
    response: (options) => {
      const data: IResourceTreeNodeEntity[] = [
        ...branchData.map((item) => ({ ...item, type: "branch" as const })),
        ...resourceData.map((item) => ({ ...item, type: "leaf" as const })),
      ];

      return {
        flag: "success",
        message: "成功",
        result: data,
      };
    },
  },
  {
    url: "/resource/get",
    type: "post",
    response: (options) => {
      const requestParams: IResourceEntity = JSON.parse(options.body);
      let entity = resourceData.find((item) => item.id == requestParams.id);
      entity = entity ? entity : resourceData[0];

      return {
        flag: "success",
        message: "",
        result: entity,
      };
    },
  },
  {
    url: "/resource/upload",
    type: "post",
    response: (options) => {
      const formData = new FormData();
      formData.append("file", options.body);

      //const file = formData.get("file") as File;
      const file = JSON.parse(options.body);

      const fileName = file ? file.name : "unknown";

      return {
        flag: "success",
        message: "",
        result: {
          id: Random.guid(),
          name: fileName,

          mime: file.type,
          url: file.type?.includes("image")
            ? resoures.image.animal.cat[0]
            : resoures.video.animal.cat[0],
        },
      };
    },
  },
  {
    url: "/resource/save",
    type: "post",
    response: (options) => {
      const entity: IResourceEntity = JSON.parse(options.body);


       const findIndex = resourceData.findIndex((item) => item.id == entity.id);

      if (findIndex == -1) {
        entity.id = Random.guid();
        resourceData.push(entity);
      } else {
        resourceData[findIndex] = entity;
      }

      return {
        flag: "success",
        message: "",
        result: entity
      };
    },
  },

  {
    url: "/resource/delete",
    type: "post",
    response: (options) => {
      const entity: IResourceEntity = JSON.parse(options.body);

      const findIndex = resourceData.findIndex((item) => item.id == entity.id);

      if (findIndex == -1) {
        return {
          flag: "error",
          message: "could not find this item",
          result: {},
        };
      } else {
        resourceData.splice(findIndex, 1);
      }
      return {
        flag: "success",
        message: "",
        result: {},
      };
    },
  },
];

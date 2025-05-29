import Mock, { MockjsMock, templateOrFn } from "mockjs";
import { IResourceEntity } from "../../../../../pages/Layout/Backend/Resource/entity";

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

interface MockItem {
  url: string | RegExp;
  type: "get" | "post";
  response: templateOrFn;
}

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
      var params = JSON.parse(options.body);
      console.log("/node/tree.............");

      return {
        flag: "success",
        message: "成功",
        result: [
          // {
          //   id: "image",
          //   parentId: "",
          //   name: "image",
          //   isLeaf: false,

          //   children: [
          //     {
          //       id: "animal",
          //       parentId: "image",
          //       name: "animal",
          //       isLeaf: false,
          //       children: [
          //         {
          //           id: "image-cat",
          //           name: "image",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "image-cat1",
          //               parentId: "image-cat",
          //               name: "cat1",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.animal.cat[0],
          //             },
          //             {
          //               id: "image-cat2",
          //               parentId: "image-cat",
          //               name: "cat2",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.animal.cat[1],
          //             },
          //           ],
          //         },
          //         {
          //           id: "dog",
          //           name: "dog",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "image-dog1",
          //               parentId: "image-dog",
          //               name: "dog1",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.animal.dog[0],
          //             },
          //             {
          //               id: "image-dog",
          //               parentId: "image-dog",
          //               name: "dog2",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.animal.dog[1],
          //             },
          //           ],
          //         },
          //       ],
          //     },
          //     {
          //       id: "image-person",
          //       parentId: "image",
          //       name: "person",
          //       isLeaf: false,
          //       children: [
          //         {
          //           id: "image-man",
          //           parentId: "image-person",
          //           name: "man",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "image-man1",
          //               name: "man1",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.people.man[0],
          //             },
          //             {
          //               id: "image-man2",
          //               name: "man2",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.people.man[1],
          //             },
          //           ],
          //         },
          //         {

          //           id: "woman",
          //           name: "image-woman",
          //           parentId: "image-person",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "image-woman1",
          //               parentId: "image-woman",
          //               name: "woman1",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.people.woman[0],
          //             },
          //             {
          //               id: "image-woman2",
          //               parentId: "image-woman",
          //               name: "woman2",
          //               isLeaf: true,
          //               mime: "image",
          //               url: resoures.image.people.woman[1],
          //             },
          //           ],
          //         },
          //       ],
          //     },
          //   ],
          // },

          //  {
          //   id: "video",
          //   parentId: "",
          //   name: "video",
          //   isLeaf: false,
          //   children: [
          //     {
          //       id: "video-animal",
          //       parentId: "video",
          //       name: "animal",
          //       isLeaf: false,
          //       children: [
          //         {
          //           id: "video-cat",
          //           parentId: "video-animal",
          //           name: "cat",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "video-cat1",
          //               parentId: "video-cat",
          //               name: "cat1",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.animal.cat[0],
          //             },
          //             {
          //               id: "video-cat2",
          //               parentId: "video-cat",
          //               name: "cat2",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.animal.cat[1],
          //             },
          //           ],
          //         },
          //         {
          //           id: "video-dog",
          //           parentId: "video-animal",
          //           name: "dog",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "video-dog1",
          //               parentId: "video-dog",
          //               name: "dog1",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.animal.dog[0],
          //             },
          //             {
          //               id: "video-dog2",
          //               parentId: "video-dog",
          //               name: "dog2",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.animal.dog[1],
          //             },
          //           ],
          //         },
          //       ],
          //     },
          //     {
          //       id: "person",
          //       parentId: "video",
          //       name: "person",
          //       isLeaf: false,
          //       children: [
          //         {
          //           id: "video-man",
          //           parentId: "person",
          //           name: "man",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "video-man1",
          //               parentId: "video-man",
          //               name: "man1",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.person.man[0],
          //             },
          //             {
          //               id: "video-man2",
          //               parentId: "video-man",
          //               name: "man2",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.person.man[1],
          //             },
          //           ],
          //         },
          //         {
          //           id: "woman",
          //           parentId: "video-person",
          //           name: "woman",
          //           isLeaf: false,
          //           children: [
          //             {
          //               id: "video-woman1",
          //               parentId: "video-woman",
          //               name: "woman1",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.person.woman[0],
          //             },
          //             {
          //               id: "video-woman2",
          //               parentId: "video-woman",
          //               name: "woman2",
          //               isLeaf: true,
          //               mime: "video",
          //               url: resoures.video.person.woman[1],
          //             },
          //           ],
          //         },
          //       ],
          //     },
          //   ],
          // },
          {
            id: "image",
            parentId: "",
            name: "image",
            isLeaf: false,
          },
          {
            id: "image-animal",
            parentId: "image",
            name: "animal",
            isLeaf: false,
          },
          {
            id: "image-cat",
            parentId: "image-animal",
            name: "image",
            isLeaf: false,
          },
          {
            id: "image-cat1",
            parentId: "image-cat",
            name: "cat1",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
          },
          {
            id: "image-cat2",
            parentId: "image-cat",
            name: "cat2",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
          },
          {
            id: "image-dog",
            parentId: "image-animal",
            name: "dog",
            isLeaf: false,
          },
          {
            id: "image-dog1",
            parentId: "image-dog",
            name: "dog1",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
          },
          {
            id: "image-dog2",
            parentId: "image-dog",
            name: "dog2",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/551628/pexels-photo-551628.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
          },
          {
            id: "image-person",
            parentId: "image",
            name: "person",
            isLeaf: false,
          },
          {
            id: "image-man",
            parentId: "image-person",
            name: "man",
            isLeaf: false,
          },
          {
            id: "image-man1",
            parentId: "image-man",
            name: "man1",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/874158/pexels-photo-874158.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
          },
          {
            id: "image-man2",
            parentId: "image-man",
            name: "man2",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/845434/pexels-photo-845434.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
          },
          {
            id: "image-woman",
            name: "image-woman",
            parentId: "image-person",
            isLeaf: false,
          },
          {
            id: "image-woman1",
            parentId: "image-woman",
            name: "woman1",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
          },
          {
            id: "image-woman2",
            parentId: "image-woman",
            name: "woman2",
            isLeaf: true,
            mime: "image",
            url: "https://images.pexels.com/photos/1130626/pexels-photo-1130626.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500",
          },
          {
            id: "video",
            parentId: "",
            name: "video",
            isLeaf: false,
          },
          {
            id: "video-animal",
            parentId: "video",
            name: "animal",
            isLeaf: false,
          },
          {
            id: "video-cat",
            parentId: "video-animal",
            name: "cat",
            isLeaf: false,
          },
          {
            id: "video-cat1",
            parentId: "video-cat",
            name: "cat1",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/1481903/1481903-sd_640_360_25fps.mp4",
          },
          {
            id: "video-cat2",
            parentId: "video-cat",
            name: "cat2",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/3116737/3116737-sd_640_360_25fps.mp4",
          },
          {
            id: "video-dog",
            parentId: "video-animal",
            name: "dog",
            isLeaf: false,
          },
          {
            id: "video-dog1",
            parentId: "video-dog",
            name: "dog1",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/3042473/3042473-sd_640_360_30fps.mp4",
          },
          {
            id: "video-dog2",
            parentId: "video-dog",
            name: "dog2",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/853770/853770-sd_640_360_25fps.mp4",
          },
          {
            id: "video-person",
            parentId: "video",
            name: "person",
            isLeaf: false,
          },
          {
            id: "video-man",
            parentId: "video-person",
            name: "man",
            isLeaf: false,
          },
          {
            id: "video-man1",
            parentId: "video-man",
            name: "man1",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/3209176/3209176-sd_640_360_25fps.mp4",
          },
          {
            id: "video-man2",
            parentId: "video-man",
            name: "man2",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/3126361/3126361-sd_640_360_25fps.mp4",
          },
          {
            id: "video-woman",
            parentId: "video-person",
            name: "woman",
            isLeaf: false,
          },
          {
            id: "video-woman1",
            parentId: "video-woman",
            name: "woman1",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/2795406/2795406-sd_640_360_25fps.mp4",
          },
          {
            id: "video-woman2",
            parentId: "video-woman",
            name: "woman2",
            isLeaf: true,
            mime: "video",
            url: "https://videos.pexels.com/video-files/3048162/3048162-sd_640_360_24fps.mp4",
          },
        ],
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
      const resource: IResourceEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {
          ...resource,
          id: resource.id ? resource.id : Random.guid(),
          name: resource.name,
          isLeaf: true,
          mime: resource.mime,
          url: resource.url,
        },
      };
    },
  },

  {
    url: "/resource/delete",
    type: "post",
    response: (options) => {
      const resource: IResourceEntity = JSON.parse(options.body);

      return {
        flag: "success",
        message: "",
        result: {},
      };
    },
  },
];

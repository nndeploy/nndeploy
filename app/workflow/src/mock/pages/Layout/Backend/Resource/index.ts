import Mock from "mockjs";

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

export default [
  {
    url: "/resource/tree",
    type: "get",
    response: (request: any) => {
      var params = JSON.parse(request.body);
      console.log("/node/tree.............");
      return {
        flag: "success",
        message: "成功",
        result: [
          {
            key: "image",
            label: "image",
            type: "branch",
            children: [
              {
                key: "animal",
                label: "animal",
                type: "branch",
                children: [
                  {
                    key: "image-cat",
                    label: "image",
                    type: "branch",
                    children: [
                      {
                        key: "image-cat1",
                        label: "cat1",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.animal.cat[0],
                      },
                      {
                        key: "image-cat2",
                        label: "cat2",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.animal.cat[1],
                      },
                    ],
                  },
                  {
                    key: "dog",
                    label: "dog",
                    type: "branch",
                    children: [
                      {
                        key: "image-dog1",
                        label: "dog1",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.animal.dog[0],
                      },
                      {
                        key: "image-dog",
                        label: "dog2",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.animal.dog[1],
                      },
                    ],
                  },
                ],
              },
              {
                key: "image-person",
                label: "person",
                type: "branch",
                children: [
                  {
                    key: "image-man",
                    label: "man",
                    type: "branch",
                    children: [
                      {
                        key: "image-man1",
                        label: "man1",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.people.man[0],
                      },
                      {
                        key: "image-man2",
                        label: "man2",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.people.man[1],
                      },
                    ],
                  },
                  {
                    label: "image-woman",
                    key: "woman",
                    type: "branch",
                    children: [
                      {
                        key: "image-woman1",
                        label: "woman1",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.people.woman[0],
                      },
                      {
                        key: "image-woman2",
                        label: "woman2",
                        type: "leaf",
                        mime: "image",
                        url: resoures.image.people.woman[1],
                      },
                    ],
                  },
                ],
              },
            ],
          },

           {
            key: "video",
            label: "video",
            type: "branch",
            children: [
              {
                key: "video-animal",
                label: "animal",
                type: "branch",
                children: [
                  {
                    key: "video-cat",
                    label: "cat",
                    type: "branch",
                    children: [
                      {
                        key: "video-cat1",
                        label: "cat1",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.animal.cat[0],
                      },
                      {
                        key: "video-cat2",
                        label: "cat2",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.animal.cat[1],
                      },
                    ],
                  },
                  {
                    key: "video-dog",
                    label: "dog",
                    type: "branch",
                    children: [
                      {
                        key: "video-dog1",
                        label: "dog1",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.animal.dog[0],
                      },
                      {
                        key: "video-dog2",
                        label: "dog2",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.animal.dog[1],
                      },
                    ],
                  },
                ],
              },
              {
                key: "person",
                label: "person",
                type: "branch",
                children: [
                  {
                    key: "video-man",
                    label: "man",
                    type: "branch",
                    children: [
                      {
                        key: "video-man1",
                        label: "man1",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.person.man[0],
                      },
                      {
                        key: "video-man2",
                        label: "man2",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.person.man[1],
                      },
                    ],
                  },
                  {
                    key: "woman",
                    label: "woman",
                    type: "branch",
                    children: [
                      {
                        key: "video-woman1",
                        label: "woman1",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.person.woman[0],
                      },
                      {
                        key: "video-woman2",
                        label: "woman2",
                        type: "leaf",
                        mime: "video",
                        url: resoures.video.person.woman[1],
                      },
                    ],
                  },
                ],
              },
            ],
          },
        ],
      };
    },
  },
];

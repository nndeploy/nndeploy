declare module '*.svg'
declare module '*.png'
declare module '*.jpg'

declare module '*.scss' {
  const content: { [className: string]: string };
  export default content;
}

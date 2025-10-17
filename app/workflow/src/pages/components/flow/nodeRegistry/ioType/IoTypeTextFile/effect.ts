import { useEffect, useState } from "react";
import { apiGetFileContent } from "./api";

export function useGetFileContent(filePath:string){

  const [fileContent, setFileContent] = useState('');

  async function getFileContent(filePath:string){

    // const response = await fetch(filePath);
    // const text = await response.text();

    const text = await apiGetFileContent(filePath)
    setFileContent(text);
  }

  useEffect(()=>{
    if(!filePath){
      return
    }

    getFileContent(filePath)

  }, [filePath])

  return fileContent

}
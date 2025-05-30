import React, { useState } from "react";
import { Popconfirm, Input, Button } from "@douyinfe/semi-ui";

interface PopconfirmWithInputProps {
  title: string; 
  //content: React.ReactNode;
  onConfirm: (value:string) => void;
  children: React.ReactNode;

}
export const PopconfirmWithInput: React.FC<PopconfirmWithInputProps> = (props) => {

  const {title, children} = props
  const [inputValue, setInputValue] = useState("");
  const [confirmDisabled, setConfirmDisabled] = useState(true);

  const handleInputChange = (value:string, e: React.ChangeEvent<HTMLInputElement>) => {
    //const value = e.target.value;
    setInputValue(value);
    setConfirmDisabled(value.trim() === "");
  };

  const handleConfirm = () => {
    //console.log("Input Value:", inputValue);
    // 在这里处理输入框的值
    props.onConfirm(inputValue)
  };

  return (
    <Popconfirm
    zIndex={2000}
      title={title}
      content={
        <Input
          placeholder="Enter something"
          value={inputValue}
          ///@ts-ignore
          onChange={handleInputChange}
          okButtonProps={{ disabled: confirmDisabled }}
        />
      }
      onConfirm={handleConfirm}
    >
      {children}
    </Popconfirm>
  );
};

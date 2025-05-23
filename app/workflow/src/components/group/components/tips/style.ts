import styled from 'styled-components';

export const GroupTipsStyle = styled.div`
  position: absolute;
  top: 35px;

  width: 100%;
  height: 28px;
  white-space: nowrap;
  pointer-events: auto;

  .container {
    display: inline-flex;
    justify-content: center;
    height: 100%;
    width: 100%;
    background-color: rgb(255 255 255);
    border-radius: 8px 8px 0 0;

    .content {
      overflow: hidden;
      display: inline-flex;
      align-items: center;
      justify-content: flex-start;

      width: fit-content;
      height: 100%;
      padding: 0 12px;

      .text {
        font-size: 14px;
        font-weight: 400;
        font-style: normal;
        line-height: 20px;
        color: rgba(15, 21, 40, 82%);
        text-overflow: ellipsis;
        margin: 0;
      }

      .space {
        width: 128px;
      }
    }

    .actions {
      display: flex;
      gap: 8px;
      align-items: center;

      height: 28px;
      padding: 0 12px;

      .close-forever {
        cursor: pointer;

        padding: 0 3px;

        font-size: 12px;
        font-weight: 400;
        font-style: normal;
        line-height: 12px;
        color: rgba(32, 41, 69, 62%);
        margin: 0;
      }

      .close {
        display: flex;
        cursor: pointer;
        height: 100%;
        align-items: center;
      }
    }
  }
`;

import "./app.tsx";

import { ReactNode } from "react";
import type { ReactNode as ReactNodeType } from "react";

function Root(): ReactNodeType {
  return null;
}

export default function main() {
  return (
    <ReactNode>
      <Root />
    </ReactNode>
  );
}
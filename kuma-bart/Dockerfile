FROM node:20-bookworm-slim

WORKDIR /workspace

COPY package.json .
RUN npm install

COPY tsconfig.json vite.config.ts vite.demo.config.ts vitest.config.ts .
COPY src/ src/
COPY demo/ demo/
COPY test/ test/

CMD ["npm", "run", "test"]
